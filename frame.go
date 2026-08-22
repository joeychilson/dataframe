package dataframe

import (
	"fmt"
	"hash/maphash"
	"reflect"
	"slices"
	"strings"

	"github.com/joeychilson/dataframe/internal/hashmap"
	"github.com/joeychilson/dataframe/mask"
	"github.com/joeychilson/dataframe/series"
)

// Field describes one column in a Frame schema. Nullable is a schema property,
// not merely a report that the current data happens to contain a null.
type Field struct {
	// Name is the column's unique, non-empty name.
	Name string
	// Type is the column's exact Go element type.
	Type reflect.Type
	// Nullable reports whether the column's schema permits null cells.
	Nullable bool
}

// ColumnSpec is the sealed heterogeneous construction value accepted by New
// and Grouped.Result. Construct values with Column or ColumnFromSeries.
type ColumnSpec interface {
	dataframeColumnSpec()
	columnName() string
	columnType() reflect.Type
	columnNullable() bool
	columnLen() int
	columnAt(int) (any, bool)
	columnRename(string) ColumnSpec
	columnTake([]int) ColumnSpec
	columnTakeNullable(series.Series[int]) ColumnSpec
	columnSlice(int, int) ColumnSpec
	columnFilter(mask.Mask) ColumnSpec
	columnConcat([]ColumnSpec) (ColumnSpec, error)
}

type columnSpec[T any] struct {
	name   string
	values series.Series[T]
}

func (columnSpec[T]) dataframeColumnSpec() {}

// Column returns a named, non-null construction column containing a shallow
// copy of values. New validates its name and row count.
func Column[T any](name string, values []T) ColumnSpec {
	return columnSpec[T]{name: name, values: series.New(values)}
}

// ColumnFromSeries returns a named construction column sharing immutable
// values. New validates its name and row count.
func ColumnFromSeries[T any](name string, values series.Series[T]) ColumnSpec {
	return columnSpec[T]{name: name, values: values}
}

// Frame is an immutable, ordered collection of equal-length named columns. Its
// zero value is a frame with zero rows and zero columns. Immutability is
// shallow: Frame and Series operations never mutate element values, but
// reference-like values stored in cells remain shared with callers.
//
// Frames retain their row count even when they contain zero columns, so Drop
// and Select can produce a zero-width frame without losing rows.
type Frame struct {
	columns  []ColumnSpec
	rowCount int
}

// New returns a Frame containing columns. It reports empty or duplicate names
// and row-count mismatches.
func New(columns ...ColumnSpec) (Frame, error) {
	if len(columns) == 0 {
		return Frame{}, nil
	}

	rowCount := -1
	names := make(map[string]struct{}, len(columns))
	result := Frame{columns: slices.Clone(columns)}
	for i, column := range result.columns {
		if column == nil {
			return Frame{}, fmt.Errorf("%w: column %d is nil", ErrSchemaMismatch, i)
		}
		name := column.columnName()
		if name == "" {
			return Frame{}, fmt.Errorf("%w: column %d", ErrInvalidName, i)
		}
		if _, exists := names[name]; exists {
			return Frame{}, fmt.Errorf("%w: %q", ErrColumnConflict, name)
		}
		names[name] = struct{}{}
		if rowCount < 0 {
			rowCount = column.columnLen()
		} else if column.columnLen() != rowCount {
			return Frame{}, fmt.Errorf("%w: column %q has %d rows, want %d", ErrRowCount, name, column.columnLen(), rowCount)
		}
	}
	result.rowCount = rowCount
	return result, nil
}

// Len returns the number of rows, including for a zero-width Frame.
func (f Frame) Len() int {
	return f.rowCount
}

// Width returns the number of columns.
func (f Frame) Width() int {
	return len(f.columns)
}

// Names returns column names in schema order.
func (f Frame) Names() []string {
	names := make([]string, len(f.columns))
	for i, column := range f.columns {
		names[i] = column.columnName()
	}
	return names
}

// Schema returns a snapshot of fields in schema order.
func (f Frame) Schema() []Field {
	fields := make([]Field, len(f.columns))
	for i, column := range f.columns {
		fields[i] = Field{
			Name:     column.columnName(),
			Type:     column.columnType(),
			Nullable: column.columnNullable(),
		}
	}
	return fields
}

// Has reports whether name exists.
func (f Frame) Has(name string) bool {
	return f.columnIndex(name) >= 0
}

// String returns a compact debugging preview. It implements fmt.Stringer; its
// output is not a stable interchange format.
func (f Frame) String() string {
	var result strings.Builder
	fmt.Fprintf(&result, "Frame[%dx%d]{", f.Len(), f.Width())
	for i, column := range f.columns {
		if i > 0 {
			result.WriteString(", ")
		}
		fmt.Fprintf(&result, "%s:%v", column.columnName(), column.columnType())
		if column.columnNullable() {
			result.WriteByte('?')
		}
	}
	result.WriteByte('}')
	return result.String()
}

var _ fmt.Stringer = Frame{}

// Column returns name as a Series with the exact requested Go element type.
//
// Errors: ErrColumnNotFound, ErrColumnType.
func (f Frame) Column[T any](name string) (series.Series[T], error) {
	index := f.columnIndex(name)
	if index < 0 {
		return series.Series[T]{}, fmt.Errorf("%w: %q", ErrColumnNotFound, name)
	}
	column := f.columns[index]
	want := reflect.TypeFor[T]()
	if column.columnType() != want {
		return series.Series[T]{}, fmt.Errorf("%w: column %q has type %v, want %v", ErrColumnType, name, column.columnType(), want)
	}
	return typedSeriesFromColumn[T](column)
}

// With adds values under name or replaces the existing column in place.
//
// Errors: ErrInvalidName, ErrRowCount.
func (f Frame) With[T any](name string, values series.Series[T]) (Frame, error) {
	if name == "" {
		return Frame{}, fmt.Errorf("%w", ErrInvalidName)
	}
	if f.Width() != 0 || f.Len() != 0 {
		if values.Len() != f.Len() {
			return Frame{}, fmt.Errorf("%w: column %q has %d rows, want %d", ErrRowCount, name, values.Len(), f.Len())
		}
	}

	column := columnSpec[T]{name: name, values: values}
	columns := slices.Clone(f.columns)
	if index := f.columnIndex(name); index >= 0 {
		columns[index] = column
	} else {
		columns = append(columns, column)
	}
	rowCount := f.Len()
	if f.Width() == 0 && f.Len() == 0 {
		rowCount = values.Len()
	}
	return Frame{columns: columns, rowCount: rowCount}, nil
}

// WithValues adds a shallow copy of values under name or replaces the existing
// column in place.
//
// Errors: ErrInvalidName, ErrRowCount.
func (f Frame) WithValues[T any](name string, values []T) (Frame, error) {
	return f.With(name, series.New(values))
}

// Drop removes names and retains the positions of all remaining columns.
// Unknown names return ErrColumnNotFound. Dropping every column retains f's row
// count.
func (f Frame) Drop(names ...string) (Frame, error) {
	if len(names) == 0 {
		return f, nil
	}
	drop := make(map[string]struct{}, len(names))
	for _, name := range names {
		if f.columnIndex(name) < 0 {
			return Frame{}, fmt.Errorf("%w: %q", ErrColumnNotFound, name)
		}
		drop[name] = struct{}{}
	}
	columns := make([]ColumnSpec, 0, len(f.columns)-len(drop))
	for _, column := range f.columns {
		if _, found := drop[column.columnName()]; !found {
			columns = append(columns, column)
		}
	}
	return Frame{columns: columns, rowCount: f.Len()}, nil
}

// Rename changes from to to while retaining column position.
//
// Errors: ErrColumnNotFound, ErrInvalidName, ErrColumnConflict.
func (f Frame) Rename(from, to string) (Frame, error) {
	index := f.columnIndex(from)
	if index < 0 {
		return Frame{}, fmt.Errorf("%w: %q", ErrColumnNotFound, from)
	}
	if to == "" {
		return Frame{}, fmt.Errorf("%w", ErrInvalidName)
	}
	if from == to {
		return f, nil
	}
	if f.Has(to) {
		return Frame{}, fmt.Errorf("%w: %q", ErrColumnConflict, to)
	}
	columns := slices.Clone(f.columns)
	columns[index] = columns[index].columnRename(to)
	return Frame{columns: columns, rowCount: f.Len()}, nil
}

// Select returns the named columns in the supplied order. With no names it
// returns a zero-width Frame retaining f.Len() rows.
//
// Errors: ErrColumnNotFound, ErrColumnConflict for duplicate selections.
func (f Frame) Select(names ...string) (Frame, error) {
	columns := make([]ColumnSpec, len(names))
	selected := make(map[string]struct{}, len(names))
	for i, name := range names {
		if _, exists := selected[name]; exists {
			return Frame{}, fmt.Errorf("%w: %q selected more than once", ErrColumnConflict, name)
		}
		index := f.columnIndex(name)
		if index < 0 {
			return Frame{}, fmt.Errorf("%w: %q", ErrColumnNotFound, name)
		}
		selected[name] = struct{}{}
		columns[i] = f.columns[index]
	}
	return Frame{columns: columns, rowCount: f.Len()}, nil
}

// Take returns requested rows in order. It panics on an invalid row index.
func (f Frame) Take(rows []int) Frame {
	for _, row := range rows {
		if row < 0 || row >= f.Len() {
			panic("dataframe: Take: row index out of range")
		}
	}
	columns := make([]ColumnSpec, len(f.columns))
	for i, column := range f.columns {
		columns[i] = column.columnTake(rows)
	}
	return Frame{columns: columns, rowCount: len(rows)}
}

// Head returns the first min(n, Len()) rows. It panics when n is negative.
func (f Frame) Head(n int) Frame {
	if n < 0 {
		panic("dataframe: Head: negative count")
	}
	return f.Slice(0, min(n, f.Len()))
}

// Tail returns the last min(n, Len()) rows. It panics when n is negative.
func (f Frame) Tail(n int) Frame {
	if n < 0 {
		panic("dataframe: Tail: negative count")
	}
	count := min(n, f.Len())
	return f.Slice(f.Len()-count, f.Len())
}

// Slice returns rows in [start, end) and shares storage with f. It panics on
// invalid bounds, like slicing a Go slice.
func (f Frame) Slice(start, end int) Frame {
	if start < 0 || end < start || end > f.Len() {
		panic("dataframe: Slice: bounds out of range")
	}
	columns := make([]ColumnSpec, len(f.columns))
	for i, column := range f.columns {
		columns[i] = column.columnSlice(start, end)
	}
	return Frame{columns: columns, rowCount: end - start}
}

// Filter keeps rows selected by selection. It panics on length mismatch.
func (f Frame) Filter(selection mask.Mask) Frame {
	if selection.Len() != f.Len() {
		panic(fmt.Sprintf("dataframe: Filter: length mismatch: frame=%d mask=%d", f.Len(), selection.Len()))
	}
	columns := make([]ColumnSpec, len(f.columns))
	for i, column := range f.columns {
		columns[i] = column.columnFilter(selection)
	}
	return Frame{columns: columns, rowCount: selection.Count()}
}

// FilterFunc keeps rows whose present values satisfy keep. Nulls are dropped
// without calling keep. It panics on length mismatch.
func (f Frame) FilterFunc[T any](values series.Series[T], keep func(T) bool) Frame {
	if values.Len() != f.Len() {
		panic(fmt.Sprintf("dataframe: FilterFunc: length mismatch: frame=%d series=%d", f.Len(), values.Len()))
	}
	selection := mask.NewFunc(f.Len(), func(i int) bool {
		value, present := values.At(i)
		return present && keep(value)
	})
	return f.Filter(selection)
}

// Distinct keeps the first row for each distinct complete row. It returns
// ErrUnsupported when any participating column cannot be compared using its
// natural Go equality. Nulls compare equal to nulls.
func (f Frame) Distinct() (Frame, error) {
	if f.Len() < 2 {
		return f, nil
	}
	if f.Width() == 0 {
		return f.Slice(0, 1), nil
	}

	fields := make([]reflect.StructField, 0, f.Width()*2)
	for i, column := range f.columns {
		if !column.columnType().Comparable() {
			return Frame{}, fmt.Errorf("%w: column %q has type %v", ErrUnsupported, column.columnName(), column.columnType())
		}
		fields = append(fields,
			reflect.StructField{Name: fmt.Sprintf("Value%d", i), Type: column.columnType()},
			reflect.StructField{Name: fmt.Sprintf("Valid%d", i), Type: reflect.TypeFor[bool]()},
		)
	}
	keyType := reflect.StructOf(fields)
	key := reflect.New(keyType).Elem()
	seen := make(map[any]struct{}, f.Len())
	rows := make([]int, 0, f.Len())
	for row := 0; row < f.Len(); row++ {
		for columnIndex, column := range f.columns {
			value, present := column.columnAt(row)
			valueField := key.Field(columnIndex * 2)
			key.Field(columnIndex*2 + 1).SetBool(present)
			if !present || value == nil {
				valueField.SetZero()
				continue
			}
			valueOf := reflect.ValueOf(value)
			if !valueOf.Comparable() {
				return Frame{}, fmt.Errorf("%w: column %q row %d contains incomparable dynamic type %v", ErrUnsupported, column.columnName(), row, valueOf.Type())
			}
			valueField.Set(valueOf)
		}
		value := key.Interface()
		if _, exists := seen[value]; exists {
			continue
		}
		seen[value] = struct{}{}
		rows = append(rows, row)
	}
	return f.Take(rows), nil
}

// DistinctBy keeps the first row for each distinct comparable positional key.
// Composite comparable structs provide a Go-native multi-column key. It panics
// on length mismatch.
func (f Frame) DistinctBy[K comparable](key series.Series[K]) Frame {
	if key.Len() != f.Len() {
		panic(fmt.Sprintf("dataframe: DistinctBy: length mismatch: frame=%d key=%d", f.Len(), key.Len()))
	}
	seen := make(map[K]struct{}, key.Len())
	nullSeen := false
	rows := make([]int, 0, key.Len())
	for i := 0; i < key.Len(); i++ {
		value, present := key.At(i)
		if !present {
			if nullSeen {
				continue
			}
			nullSeen = true
			rows = append(rows, i)
			continue
		}
		if _, exists := seen[value]; exists {
			continue
		}
		seen[value] = struct{}{}
		rows = append(rows, i)
	}
	return f.Take(rows)
}

// DistinctByUsing is DistinctBy for non-comparable keys or custom equality. It
// panics on length mismatch or a nil hasher.
func (f Frame) DistinctByUsing[K any](key series.Series[K], hasher maphash.Hasher[K]) Frame {
	if key.Len() != f.Len() {
		panic(fmt.Sprintf("dataframe: DistinctByUsing: length mismatch: frame=%d key=%d", f.Len(), key.Len()))
	}
	seen := hashmap.New[K, struct{}](hasher)
	nullSeen := false
	rows := make([]int, 0, key.Len())
	for i := 0; i < key.Len(); i++ {
		value, present := key.At(i)
		if !present {
			if nullSeen {
				continue
			}
			nullSeen = true
			rows = append(rows, i)
			continue
		}
		if _, loaded := seen.LoadOrStore(value, struct{}{}); loaded {
			continue
		}
		rows = append(rows, i)
	}
	return f.Take(rows)
}

// Concat appends others' rows below f. All frames must have identical column
// names, order, and exact Go types. Output nullability is widened when needed.
// Zero-width frames concatenate by adding their retained row counts.
//
// Errors: ErrSchemaMismatch.
func (f Frame) Concat(others ...Frame) (Frame, error) {
	if len(others) == 0 {
		return f, nil
	}
	maxInt := int(^uint(0) >> 1)
	total := f.Len()
	for frameIndex, other := range others {
		if other.Width() != f.Width() {
			return Frame{}, fmt.Errorf("%w: frame %d has %d columns, want %d", ErrSchemaMismatch, frameIndex+1, other.Width(), f.Width())
		}
		for columnIndex, column := range f.columns {
			otherColumn := other.columns[columnIndex]
			if otherColumn.columnName() != column.columnName() || otherColumn.columnType() != column.columnType() {
				return Frame{}, fmt.Errorf("%w: frame %d column %d is %q:%v, want %q:%v", ErrSchemaMismatch, frameIndex+1, columnIndex, otherColumn.columnName(), otherColumn.columnType(), column.columnName(), column.columnType())
			}
		}
		if other.Len() > maxInt-total {
			return Frame{}, fmt.Errorf("%w: concatenated row count overflows int", ErrRowCount)
		}
		total += other.Len()
	}

	columns := make([]ColumnSpec, f.Width())
	for columnIndex, column := range f.columns {
		parts := make([]ColumnSpec, len(others))
		for i, other := range others {
			parts[i] = other.columns[columnIndex]
		}
		joined, err := column.columnConcat(parts)
		if err != nil {
			return Frame{}, err
		}
		columns[columnIndex] = joined
	}
	return Frame{columns: columns, rowCount: total}, nil
}

func (c columnSpec[T]) columnName() string {
	return c.name
}

func (c columnSpec[T]) columnType() reflect.Type {
	return reflect.TypeFor[T]()
}

func (c columnSpec[T]) columnNullable() bool {
	return c.values.Nullable()
}

func (c columnSpec[T]) columnLen() int {
	return c.values.Len()
}

func (c columnSpec[T]) columnAt(i int) (any, bool) {
	value, present := c.values.At(i)
	if !present {
		return nil, false
	}
	return value, true
}

func (c columnSpec[T]) columnRename(name string) ColumnSpec {
	c.name = name
	return c
}

func (c columnSpec[T]) columnTake(rows []int) ColumnSpec {
	c.values = c.values.Take(rows)
	return c
}

func (c columnSpec[T]) columnTakeNullable(rows series.Series[int]) ColumnSpec {
	c.values = c.values.TakeNullable(rows)
	return c
}

func (c columnSpec[T]) columnSlice(start, end int) ColumnSpec {
	c.values = c.values.Slice(start, end)
	return c
}

func (c columnSpec[T]) columnFilter(selection mask.Mask) ColumnSpec {
	c.values = c.values.Filter(selection)
	return c
}

func (c columnSpec[T]) columnConcat(others []ColumnSpec) (ColumnSpec, error) {
	parts := make([]series.Series[T], len(others))
	for i, other := range others {
		if other.columnType() != reflect.TypeFor[T]() {
			return nil, fmt.Errorf("%w: column %q type %v does not match %v", ErrSchemaMismatch, c.name, other.columnType(), reflect.TypeFor[T]())
		}
		values, err := typedSeriesFromColumn[T](other)
		if err != nil {
			return nil, err
		}
		parts[i] = values
	}
	c.values = c.values.Concat(parts...)
	return c, nil
}

func (f Frame) columnIndex(name string) int {
	for i, column := range f.columns {
		if column.columnName() == name {
			return i
		}
	}
	return -1
}
