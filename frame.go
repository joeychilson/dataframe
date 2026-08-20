package dataframe

import (
	"cmp"
	"errors"
	"fmt"
	"reflect"
	"slices"

	"github.com/joeychilson/dataframe/series"
)

var (
	// ErrColumnNotFound is returned when an operation names a column that is not
	// present in a Frame.
	ErrColumnNotFound = errors.New("column not found")

	// ErrColumnType is returned when a typed operation requests a column using
	// a different Go type than the one stored in the Frame.
	ErrColumnType = errors.New("column type mismatch")

	// ErrRowCount is returned when a column's length does not match the Frame.
	ErrRowCount = errors.New("row count mismatch")

	// ErrInvalidName is returned when a column name is empty.
	ErrInvalidName = errors.New("column name must not be empty")

	// ErrGroupKey is returned when an aggregate result would replace the column
	// that identifies the groups.
	ErrGroupKey = errors.New("aggregate name conflicts with group key")

	// ErrSchemaMismatch is returned when Frames cannot be combined because their
	// column names, order, or exact Go types differ.
	ErrSchemaMismatch = errors.New("frame schema mismatch")

	// ErrColumnConflict is returned when an operation would produce duplicate
	// column names.
	ErrColumnConflict = errors.New("column name conflict")
)

// Field describes one column in a Frame schema.
type Field struct {
	Name     string
	Type     reflect.Type
	Nullable bool
}

// Frame is an immutable, ordered collection of equal-length, heterogeneous
// columns. Its zero value is an empty Frame ready for use.
type Frame struct {
	columns []namedColumn
	// index is immutable and may be shared by Frames with the same schema.
	index map[string]int
}

// New returns an empty Frame. The zero value of Frame is equivalent.
func New() Frame {
	return Frame{}
}

// Len returns the number of rows.
func (f Frame) Len() int {
	if len(f.columns) == 0 {
		return 0
	}
	return f.columns[0].data.len()
}

// Width returns the number of columns.
func (f Frame) Width() int {
	return len(f.columns)
}

// Names returns column names in schema order.
func (f Frame) Names() []string {
	names := make([]string, len(f.columns))
	for i, column := range f.columns {
		names[i] = column.name
	}
	return names
}

// Schema returns a snapshot of the Frame schema.
func (f Frame) Schema() []Field {
	fields := make([]Field, len(f.columns))
	for i, column := range f.columns {
		fields[i] = Field{
			Name:     column.name,
			Type:     column.data.columnType(),
			Nullable: column.data.columnNullable(),
		}
	}
	return fields
}

// Column returns name as a typed Series. It checks T against the exact stored
// Go type once; subsequent Series operations are fully statically typed.
func (f Frame) Column[T any](name string) (series.Series[T], error) {
	column, err := f.lookup(name)
	if err != nil {
		return series.Series[T]{}, err
	}

	typed, ok := column.(typedColumn[T])
	if !ok {
		return series.Series[T]{}, fmt.Errorf(
			"%w for %q: stored %v, requested %v",
			ErrColumnType,
			name,
			column.columnType(),
			reflect.TypeFor[T](),
		)
	}
	return typed.Series, nil
}

// WithColumn returns a Frame containing a non-nullable column named name.
// T is inferred from values. Existing columns with the same name are replaced
// in place without changing schema order.
func (f Frame) WithColumn[T any](name string, values []T) (Frame, error) {
	return f.WithSeries(name, series.New(values))
}

// WithNullableColumn is WithColumn with an explicit per-row validity slice.
func (f Frame) WithNullableColumn[T any](name string, values []T, validity []bool) (Frame, error) {
	column, err := series.NewNullable(values, validity)
	if err != nil {
		return Frame{}, fmt.Errorf("column %q: %w", name, err)
	}
	return f.WithSeries(name, column)
}

// WithSeries returns a Frame containing series under name. Frame and Series
// immutability make this a cheap, safe composition operation.
func (f Frame) WithSeries[T any](name string, column series.Series[T]) (Frame, error) {
	if name == "" {
		return Frame{}, ErrInvalidName
	}

	existing, replacing := f.position(name)
	if err := f.checkLength(name, column.Len()); err != nil {
		return Frame{}, fmt.Errorf("column %q: %w", name, err)
	}

	columns := slices.Clone(f.columns)
	stored := typedColumn[T]{Series: column}
	if replacing {
		columns[existing].data = stored
		return Frame{columns: columns, index: f.index}, nil
	}

	columns = append(columns, namedColumn{name: name, data: stored})
	return makeFrame(columns), nil
}

// Concat returns a Frame containing f's rows followed by each input Frame's
// rows in order. Column names, order, and exact Go types must match across all
// Frames. A result column is nullable when that column is nullable in any input
// Frame. Concatenating no inputs returns f unchanged.
func (f Frame) Concat(others ...Frame) (Frame, error) {
	if len(others) == 0 {
		return f, nil
	}

	for i, other := range others {
		frameIndex := i + 1
		if other.Width() != f.Width() {
			return Frame{}, fmt.Errorf(
				"%w: frame %d has %d columns, want %d",
				ErrSchemaMismatch,
				frameIndex,
				other.Width(),
				f.Width(),
			)
		}

		for column := range f.columns {
			got := other.columns[column]
			want := f.columns[column]
			if got.name != want.name {
				return Frame{}, fmt.Errorf(
					"%w: frame %d column %d is %q, want %q",
					ErrSchemaMismatch,
					frameIndex,
					column,
					got.name,
					want.name,
				)
			}
			if got.data.columnType() != want.data.columnType() {
				return Frame{}, fmt.Errorf(
					"%w: frame %d column %q has type %v, want %v",
					ErrSchemaMismatch,
					frameIndex,
					got.name,
					got.data.columnType(),
					want.data.columnType(),
				)
			}
		}
	}

	columns := make([]namedColumn, f.Width())
	for column, firstColumn := range f.columns {
		remaining := make([]columnData, len(others))
		for i, other := range others {
			remaining[i] = other.columns[column].data
		}
		columns[column] = namedColumn{
			name: firstColumn.name,
			data: firstColumn.data.concat(remaining),
		}
	}
	return Frame{columns: columns, index: f.index}, nil
}

// Derive maps source into a new or replacement target column. Both A and B are
// inferred from fn, making type-changing transforms natural method calls.
func (f Frame) Derive[A, B any](source, target string, fn func(A) B) (Frame, error) {
	series, err := f.Column[A](source)
	if err != nil {
		return Frame{}, err
	}
	return f.WithSeries(target, series.Map(fn))
}

// TryDerive is Derive for transforms that can fail. It stops at the first
// error and annotates it with the row index. Null rows are propagated without
// calling fn.
func (f Frame) TryDerive[A, B any](source, target string, fn func(A) (B, error)) (Frame, error) {
	series, err := f.Column[A](source)
	if err != nil {
		return Frame{}, err
	}

	mapped, err := series.TryMap(fn)
	if err != nil {
		return Frame{}, fmt.Errorf("derive %q from %q: %w", target, source, err)
	}
	return f.WithSeries(target, mapped)
}

// Derive2 combines corresponding present values from left and right into a new
// or replacement target column. A target row is null when either source row is
// null. A, B, and C are inferred from fn.
func (f Frame) Derive2[A, B, C any](left, right, target string, fn func(A, B) C) (Frame, error) {
	leftValues, err := f.Column[A](left)
	if err != nil {
		return Frame{}, err
	}
	rightValues, err := f.Column[B](right)
	if err != nil {
		return Frame{}, err
	}
	return f.WithSeries(target, leftValues.Map2(rightValues, fn))
}

// TryDerive2 is Derive2 for transforms that can fail. It stops at the first
// error and annotates it with the row index. Null rows are propagated without
// calling fn.
func (f Frame) TryDerive2[A, B, C any](left, right, target string, fn func(A, B) (C, error)) (Frame, error) {
	leftValues, err := f.Column[A](left)
	if err != nil {
		return Frame{}, err
	}
	rightValues, err := f.Column[B](right)
	if err != nil {
		return Frame{}, err
	}

	mapped, err := leftValues.TryMap2(rightValues, fn)
	if err != nil {
		return Frame{}, fmt.Errorf("derive %q from %q and %q: %w", target, left, right, err)
	}
	return f.WithSeries(target, mapped)
}

// Filter keeps rows whose present value in column satisfies predicate. Nulls
// in the predicate column are not retained.
func (f Frame) Filter[T any](column string, predicate func(T) bool) (Frame, error) {
	values, err := f.Column[T](column)
	if err != nil {
		return Frame{}, err
	}
	return f.take(values.MatchingRows(predicate)), nil
}

// FillNull returns a Frame in which nulls in column are replaced by value. The
// resulting column is non-nullable. T is inferred from value and must exactly
// match the stored column type.
func (f Frame) FillNull[T any](column string, value T) (Frame, error) {
	values, err := f.Column[T](column)
	if err != nil {
		return Frame{}, err
	}
	return f.WithSeries(column, values.FillNull(value))
}

// DropNull returns a Frame containing rows where column is present. Row order
// and column schemas are preserved.
func (f Frame) DropNull(column string) (Frame, error) {
	values, err := f.lookup(column)
	if err != nil {
		return Frame{}, err
	}
	if !values.columnNullable() {
		return f, nil
	}
	return f.take(values.presentRows()), nil
}

// Slice returns a Frame containing rows in the half-open range [start:end]. It
// shares immutable column storage with f and panics when the bounds would be
// invalid for a Go slice.
func (f Frame) Slice(start, end int) Frame {
	if start < 0 || end < start || end > f.Len() {
		panic(fmt.Sprintf("dataframe: slice bounds out of range [%d:%d] with length %d", start, end, f.Len()))
	}

	columns := make([]namedColumn, len(f.columns))
	for i, column := range f.columns {
		columns[i] = namedColumn{
			name: column.name,
			data: column.data.slice(start, end),
		}
	}
	return Frame{columns: columns, index: f.index}
}

// DistinctOn returns the first row for each distinct value in column. Rows
// retain first-seen order, null keys form one group distinct from the zero
// value of K, and present keys use Go equality, so each NaN is distinct.
func (f Frame) DistinctOn[K comparable](column string) (Frame, error) {
	keys, err := f.Column[K](column)
	if err != nil {
		return Frame{}, err
	}
	return f.take(series.UniqueRows(keys)), nil
}

// Join returns the inner join of f and right using a key column with the same
// name in both Frames.
func (f Frame) Join[K comparable](right Frame, key string) (Frame, error) {
	return f.JoinOn[K](right, key, key)
}

// JoinOn returns the inner join of f and right using leftKey and rightKey.
// Output columns are all left columns followed by right columns except the
// right key. Conflicting right column names are rejected. Duplicate keys
// produce every matching pair in stable left-row, then right-row, order. Keys
// use Go equality; nulls and present NaNs do not match.
func (f Frame) JoinOn[K comparable](right Frame, leftKey, rightKey string) (Frame, error) {
	leftKeys, rightKeys, extras, err := f.prepareJoin[K](right, leftKey, rightKey)
	if err != nil {
		return Frame{}, err
	}

	leftRows, rightRows := series.JoinRows(leftKeys, rightKeys)
	left := f.take(leftRows)
	columns := make([]namedColumn, 0, left.Width()+len(extras))
	columns = append(columns, left.columns...)
	for _, column := range extras {
		columns = append(columns, namedColumn{
			name: column.name,
			data: column.data.take(rightRows),
		})
	}
	return makeFrame(columns), nil
}

// LeftJoin returns the left join of f and right using a key column with the
// same name in both Frames.
func (f Frame) LeftJoin[K comparable](right Frame, key string) (Frame, error) {
	return f.LeftJoinOn[K](right, key, key)
}

// LeftJoinOn returns the left join of f and right using leftKey and rightKey.
// Output columns are all left columns followed by nullable right columns except
// the right key. Conflicting right column names are rejected. Duplicate keys
// produce every matching pair in stable left-row, then right-row, order. An
// unmatched left row appears once with null right values. Keys use Go equality;
// nulls and present NaNs do not match.
func (f Frame) LeftJoinOn[K comparable](right Frame, leftKey, rightKey string) (Frame, error) {
	leftKeys, rightKeys, extras, err := f.prepareJoin[K](right, leftKey, rightKey)
	if err != nil {
		return Frame{}, err
	}

	leftRows, rightRows := series.LeftJoinRows(leftKeys, rightKeys)
	left := f.take(leftRows)
	columns := make([]namedColumn, 0, left.Width()+len(extras))
	columns = append(columns, left.columns...)
	for _, column := range extras {
		columns = append(columns, namedColumn{
			name: column.name,
			data: column.data.takeNullable(rightRows),
		})
	}
	return makeFrame(columns), nil
}

func (f Frame) prepareJoin[K comparable](right Frame, leftKey, rightKey string) (
	series.Series[K],
	series.Series[K],
	[]namedColumn,
	error,
) {
	leftKeys, err := f.Column[K](leftKey)
	if err != nil {
		return series.Series[K]{}, series.Series[K]{}, nil, fmt.Errorf("left join key: %w", err)
	}
	rightKeys, err := right.Column[K](rightKey)
	if err != nil {
		return series.Series[K]{}, series.Series[K]{}, nil, fmt.Errorf("right join key: %w", err)
	}

	extras := make([]namedColumn, 0, right.Width()-1)
	for _, column := range right.columns {
		if column.name == rightKey {
			continue
		}
		if _, conflict := f.position(column.name); conflict {
			return series.Series[K]{}, series.Series[K]{}, nil, fmt.Errorf("%w: %q", ErrColumnConflict, column.name)
		}
		extras = append(extras, column)
	}
	return leftKeys, rightKeys, extras, nil
}

// Grouped is an immutable grouping of a Frame by a comparable column. Groups
// and aggregate rows retain the order in which their keys first appeared.
type Grouped[K comparable] struct {
	source Frame
	key    string
	rows   [][]int
	result Frame
}

// GroupBy partitions rows by the values in column. Null keys form one group,
// distinct from the zero value of K. Present keys use Go equality, so each NaN
// forms a separate group.
func (f Frame) GroupBy[K comparable](column string) (Grouped[K], error) {
	keys, err := f.Column[K](column)
	if err != nil {
		return Grouped[K]{}, err
	}

	rows := series.GroupRows(keys)
	first := make([]int, len(rows))
	for i, group := range rows {
		first[i] = group[0]
	}

	result, err := New().WithSeries(column, keys.Take(first))
	if err != nil {
		return Grouped[K]{}, err
	}
	return Grouped[K]{source: f, key: column, rows: rows, result: result}, nil
}

// WithAggregate returns a Grouped value with name containing one result per
// group. T names the exact source column type; U is inferred from aggregate.
// A false aggregate result is stored as null. Existing result columns with the
// same name are replaced in place, but the group-key name is reserved.
func (g Grouped[K]) WithAggregate[T, U any](column, name string, aggregate func(series.Series[T]) (U, bool)) (Grouped[K], error) {
	if name == "" {
		return Grouped[K]{}, ErrInvalidName
	}
	if name == g.key {
		return Grouped[K]{}, fmt.Errorf("%w: %q", ErrGroupKey, name)
	}

	values, err := g.source.Column[T](column)
	if err != nil {
		return Grouped[K]{}, err
	}

	results := make([]U, len(g.rows))
	validity := make([]bool, len(g.rows))
	for i, rows := range g.rows {
		results[i], validity[i] = aggregate(values.Take(rows))
	}

	resultSeries, err := series.NewNullable(results, validity)
	if err != nil {
		return Grouped[K]{}, fmt.Errorf("aggregate %q: %w", name, err)
	}
	result, err := g.result.WithSeries(name, resultSeries)
	if err != nil {
		return Grouped[K]{}, err
	}

	return Grouped[K]{source: g.source, key: g.key, rows: g.rows, result: result}, nil
}

// Frame returns the current grouped result. It initially contains only the
// group key, followed by columns added with WithAggregate.
func (g Grouped[K]) Frame() Frame {
	return g.result
}

// SortOptions configures value direction and null placement. Its zero value
// sorts ascending with nulls last.
type SortOptions struct {
	Descending bool
	NullsFirst bool
}

// Sort stably orders every column by an ordered column. Equal values retain
// their original row order.
func (f Frame) Sort[T cmp.Ordered](column string, options SortOptions) (Frame, error) {
	return f.SortBy(column, cmp.Compare[T], options)
}

// SortBy stably orders every column using compare for the named column. Equal
// values retain their original row order, and compare is not called for nulls.
func (f Frame) SortBy[T any](column string, compare func(T, T) int, options SortOptions) (Frame, error) {
	values, err := f.Column[T](column)
	if err != nil {
		return Frame{}, err
	}

	if options.Descending {
		ascending := compare
		compare = func(left, right T) int {
			return ascending(right, left)
		}
	}

	rows := values.SortedRows(compare, options.NullsFirst)
	return f.take(rows), nil
}

// Select returns a Frame containing names in the requested order.
func (f Frame) Select(names ...string) (Frame, error) {
	columns := make([]namedColumn, len(names))
	selected := make(map[string]struct{}, len(names))
	for i, name := range names {
		if _, duplicate := selected[name]; duplicate {
			return Frame{}, fmt.Errorf("duplicate selected column %q", name)
		}
		column, err := f.lookup(name)
		if err != nil {
			return Frame{}, err
		}
		columns[i] = namedColumn{name: name, data: column}
		selected[name] = struct{}{}
	}
	return makeFrame(columns), nil
}

// Rename returns a Frame with the column named from renamed to to in the same
// schema position. Renaming a column to itself is a no-op. The target name must
// be non-empty and must not already belong to another column.
func (f Frame) Rename(from, to string) (Frame, error) {
	if to == "" {
		return Frame{}, ErrInvalidName
	}

	position, ok := f.position(from)
	if !ok {
		return Frame{}, fmt.Errorf("%w: %q", ErrColumnNotFound, from)
	}
	if from == to {
		return f, nil
	}
	if _, conflict := f.position(to); conflict {
		return Frame{}, fmt.Errorf("%w: %q", ErrColumnConflict, to)
	}

	columns := slices.Clone(f.columns)
	columns[position].name = to
	return makeFrame(columns), nil
}

// Drop returns a Frame without name. Dropping a missing column is an error.
func (f Frame) Drop(name string) (Frame, error) {
	position, ok := f.position(name)
	if !ok {
		return Frame{}, fmt.Errorf("%w: %q", ErrColumnNotFound, name)
	}

	columns := slices.Delete(slices.Clone(f.columns), position, position+1)
	return makeFrame(columns), nil
}

func (f Frame) take(rows []int) Frame {
	columns := make([]namedColumn, len(f.columns))
	for i, column := range f.columns {
		columns[i] = namedColumn{name: column.name, data: column.data.take(rows)}
	}
	return Frame{columns: columns, index: f.index}
}

func (f Frame) lookup(name string) (columnData, error) {
	position, ok := f.position(name)
	if !ok {
		return nil, fmt.Errorf("%w: %q", ErrColumnNotFound, name)
	}
	return f.columns[position].data, nil
}

func (f Frame) position(name string) (int, bool) {
	if f.index != nil {
		position, ok := f.index[name]
		return position, ok
	}

	// Preserve the zero-value guarantee and remain robust if an internal caller
	// constructs a Frame without using makeFrame.
	for i, column := range f.columns {
		if column.name == name {
			return i, true
		}
	}
	return 0, false
}

func (f Frame) checkLength(name string, rows int) error {
	for _, column := range f.columns {
		if column.name == name {
			continue
		}
		if existing := column.data.len(); existing != rows {
			return fmt.Errorf("%w: frame has %d rows, column has %d", ErrRowCount, existing, rows)
		}
	}
	return nil
}

type columnData interface {
	columnType() reflect.Type
	columnNullable() bool
	len() int
	presentRows() []int
	slice(int, int) columnData
	take([]int) columnData
	takeNullable(series.Series[int]) columnData
	concat([]columnData) columnData
}

type namedColumn struct {
	name string
	data columnData
}

func makeFrame(columns []namedColumn) Frame {
	if len(columns) == 0 {
		return Frame{columns: columns}
	}

	index := make(map[string]int, len(columns))
	for i, column := range columns {
		index[column.name] = i
	}
	return Frame{columns: columns, index: index}
}

// typedColumn adapts the public series type to Frame's deliberately
// non-generic storage interface. The adapter keeps storage details private to
// this package without creating an import cycle.
type typedColumn[T any] struct {
	series.Series[T]
}

func (c typedColumn[T]) columnType() reflect.Type { return c.Type() }
func (c typedColumn[T]) columnNullable() bool     { return c.Nullable() }
func (c typedColumn[T]) len() int                 { return c.Len() }
func (c typedColumn[T]) presentRows() []int       { return c.PresentRows() }
func (c typedColumn[T]) slice(start, end int) columnData {
	return typedColumn[T]{Series: c.Slice(start, end)}
}
func (c typedColumn[T]) take(rows []int) columnData {
	return typedColumn[T]{Series: c.Take(rows)}
}
func (c typedColumn[T]) takeNullable(rows series.Series[int]) columnData {
	return typedColumn[T]{Series: c.TakeNullable(rows)}
}
func (c typedColumn[T]) concat(columns []columnData) columnData {
	parts := make([]series.Series[T], len(columns))
	for i, column := range columns {
		typed, ok := column.(typedColumn[T])
		if !ok {
			panic("dataframe: inconsistent column type during concatenation")
		}
		parts[i] = typed.Series
	}
	return typedColumn[T]{Series: c.Series.Concat(parts...)}
}
