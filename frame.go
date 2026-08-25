// Package dataframe provides immutable, ordered tables whose columns are
// typed series values.
//
// A Frame's schema is dynamic because column names are strings. [Frame.Column]
// is the checked boundary: it verifies the name and exact Go type, then returns
// an ordinary typed Series. [Frame.Columns] provides read-only dynamic access
// for schema-driven code. Series and masks align positionally; operations
// validate lengths whenever they combine rows.
//
// Panics are reserved for programmer-contract violations in operations whose
// signatures cannot return errors, including invalid indexes or bounds,
// incompatible positional lengths, and invalid callback or numeric arguments.
// Dynamic schema operations, construction, joins, and record conversion return
// errors.
package dataframe

import (
	"errors"
	"fmt"
	"math"
	"reflect"
	"slices"
	"strings"

	"github.com/joeychilson/dataframe/internal/record"
	"github.com/joeychilson/dataframe/mask"
	"github.com/joeychilson/dataframe/series"
)

var (
	// ErrColumnNotFound reports that a requested column name is absent.
	ErrColumnNotFound = errors.New("column not found")
	// ErrColumnType reports that a column does not have the requested exact Go type.
	ErrColumnType = errors.New("column type mismatch")
	// ErrRowCount reports incompatible positional lengths or frame row counts.
	ErrRowCount = errors.New("row count mismatch")
	// ErrInvalidName reports an empty column name.
	ErrInvalidName = errors.New("column name must not be empty")
	// ErrColumnConflict reports duplicate or colliding output column names.
	ErrColumnConflict = record.ErrColumnConflict
	// ErrSchemaMismatch reports frames whose ordered schemas cannot be combined.
	ErrSchemaMismatch = errors.New("frame schema mismatch")
	// ErrUnsupported reports a value or type that an operation cannot handle.
	ErrUnsupported = errors.New("unsupported operation for column type")
	// ErrInvalidRecord reports an unsupported record type or field value.
	ErrInvalidRecord = record.ErrInvalidRecord
	// ErrInvalidRow reports an unusable row value.
	ErrInvalidRow = errors.New("invalid row")
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
	dataframeColumnSpec() (column, int)
}

// Frame is an immutable, ordered collection of equal-length named columns. Its
// zero value is a frame with zero rows and zero columns. Immutability is
// shallow: Frame and Series operations never mutate element values, but
// reference-like values stored in cells remain shared with callers.
//
// Frames retain their row count even when they contain zero columns, so Drop
// and Select can produce a zero-width frame without losing rows.
type Frame struct {
	columns  []column
	rowCount int
}

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

// New returns a Frame containing columns. It reports empty or duplicate names
// and row-count mismatches.
func New(columns ...ColumnSpec) (Frame, error) {
	if len(columns) == 0 {
		return Frame{}, nil
	}

	rowCount := -1
	names := make(map[string]struct{}, len(columns))
	result := Frame{columns: make([]column, len(columns))}
	for i, spec := range columns {
		if spec == nil {
			return Frame{}, fmt.Errorf("%w: column %d is nil", ErrSchemaMismatch, i)
		}
		stored, length := spec.dataframeColumnSpec()
		result.columns[i] = stored
		name := stored.name
		if name == "" {
			return Frame{}, fmt.Errorf("%w: column %d", ErrInvalidName, i)
		}
		if _, exists := names[name]; exists {
			return Frame{}, fmt.Errorf("%w: %q", ErrColumnConflict, name)
		}
		names[name] = struct{}{}
		if rowCount < 0 {
			rowCount = length
		} else if length != rowCount {
			return Frame{}, fmt.Errorf("%w: column %q has %d rows, want %d", ErrRowCount, name, length, rowCount)
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
		names[i] = column.name
	}
	return names
}

// Schema returns a snapshot of fields in schema order.
func (f Frame) Schema() []Field {
	fields := make([]Field, len(f.columns))
	for i, column := range f.columns {
		fields[i] = Field{
			Name:     column.name,
			Type:     column.typeOf,
			Nullable: column.nullable,
		}
	}
	return fields
}

// Has reports whether name exists.
func (f Frame) Has(name string) bool {
	return slices.IndexFunc(f.columns, func(column column) bool {
		return column.name == name
	}) >= 0
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
		fmt.Fprintf(&result, "%s:%v", column.name, column.typeOf)
		if column.nullable {
			result.WriteByte('?')
		}
	}
	result.WriteByte('}')
	return result.String()
}

// Column returns name as a Series with the exact requested Go element type.
//
// Errors: ErrColumnNotFound, ErrColumnType.
func (f Frame) Column[T any](name string) (series.Series[T], error) {
	index := slices.IndexFunc(f.columns, func(column column) bool {
		return column.name == name
	})
	if index < 0 {
		return series.Series[T]{}, fmt.Errorf("%w: %q", ErrColumnNotFound, name)
	}
	stored := f.columns[index]
	want := reflect.TypeFor[T]()
	if stored.typeOf != want {
		return series.Series[T]{}, fmt.Errorf("%w: column %q has type %v, want %v", ErrColumnType, name, stored.typeOf, want)
	}
	return typedSeriesFromData[T](stored.values), nil
}

// With adds values under name or replaces the existing column in place.
//
// Errors: ErrInvalidName, ErrRowCount.
func (f Frame) With[T any](name string, values series.Series[T]) (Frame, error) {
	if name == "" {
		return Frame{}, ErrInvalidName
	}
	if f.Width() != 0 || f.Len() != 0 {
		if values.Len() != f.Len() {
			return Frame{}, fmt.Errorf("%w: column %q has %d rows, want %d", ErrRowCount, name, values.Len(), f.Len())
		}
	}

	newColumn := typedColumn(name, values)
	columns := slices.Clone(f.columns)
	if index := slices.IndexFunc(f.columns, func(candidate column) bool {
		return candidate.name == name
	}); index >= 0 {
		columns[index] = newColumn
	} else {
		columns = append(columns, newColumn)
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
		if !f.Has(name) {
			return Frame{}, fmt.Errorf("%w: %q", ErrColumnNotFound, name)
		}
		drop[name] = struct{}{}
	}
	columns := make([]column, 0, len(f.columns)-len(drop))
	for _, column := range f.columns {
		if _, found := drop[column.name]; !found {
			columns = append(columns, column)
		}
	}
	return Frame{columns: columns, rowCount: f.Len()}, nil
}

// Rename changes the column named from to the name to while retaining its position.
//
// Errors: ErrColumnNotFound, ErrInvalidName, ErrColumnConflict.
func (f Frame) Rename(from, to string) (Frame, error) {
	index := slices.IndexFunc(f.columns, func(column column) bool {
		return column.name == from
	})
	if index < 0 {
		return Frame{}, fmt.Errorf("%w: %q", ErrColumnNotFound, from)
	}
	if to == "" {
		return Frame{}, ErrInvalidName
	}
	if from == to {
		return f, nil
	}
	if f.Has(to) {
		return Frame{}, fmt.Errorf("%w: %q", ErrColumnConflict, to)
	}
	columns := slices.Clone(f.columns)
	columns[index].name = to
	return Frame{columns: columns, rowCount: f.Len()}, nil
}

// Select returns the named columns in the supplied order. With no names it
// returns a zero-width Frame retaining f.Len() rows.
//
// Errors: ErrColumnNotFound, ErrColumnConflict for duplicate selections.
func (f Frame) Select(names ...string) (Frame, error) {
	columns := make([]column, len(names))
	selected := make(map[string]struct{}, len(names))
	for i, name := range names {
		if _, exists := selected[name]; exists {
			return Frame{}, fmt.Errorf("%w: %q selected more than once", ErrColumnConflict, name)
		}
		index := slices.IndexFunc(f.columns, func(column column) bool {
			return column.name == name
		})
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
	identity := len(rows) == f.Len()
	for i, row := range rows {
		if row < 0 || row >= f.Len() {
			panic("dataframe: Take: row index out of range")
		}
		identity = identity && row == i
	}
	if identity {
		return f
	}
	columns := make([]column, len(f.columns))
	for i, column := range f.columns {
		columns[i] = column
		columns[i].values = column.values.take(rows)
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
	if start == 0 && end == f.Len() {
		return f
	}
	columns := make([]column, len(f.columns))
	for i, column := range f.columns {
		columns[i] = column
		columns[i].values = column.values.slice(start, end)
	}
	return Frame{columns: columns, rowCount: end - start}
}

// Filter keeps rows selected by selection. It panics on length mismatch.
func (f Frame) Filter(selection mask.Mask) Frame {
	if selection.Len() != f.Len() {
		panic(fmt.Sprintf("dataframe: Filter: length mismatch: frame=%d mask=%d", f.Len(), selection.Len()))
	}
	rowCount := selection.Count()
	if rowCount == f.Len() {
		return f
	}
	columns := make([]column, len(f.columns))
	for i, column := range f.columns {
		columns[i] = column
		columns[i].values = column.values.filter(selection)
	}
	return Frame{columns: columns, rowCount: rowCount}
}

// FilterFunc keeps rows whose present values satisfy keep. Nulls are dropped
// without calling keep. FilterFunc panics on length mismatch or when keep is
// nil.
func (f Frame) FilterFunc[T any](values series.Series[T], keep func(T) bool) Frame {
	if values.Len() != f.Len() {
		panic(fmt.Sprintf("dataframe: FilterFunc: length mismatch: frame=%d series=%d", f.Len(), values.Len()))
	}
	if keep == nil {
		panic("dataframe: FilterFunc: nil predicate")
	}
	selection := mask.NewFunc(f.Len(), func(i int) bool {
		value, present := values.At(i)
		return present && keep(value)
	})
	return f.Filter(selection)
}

// Concat appends others' rows below f. All frames must have identical column
// names, order, and exact Go types. Output nullability is widened when needed.
// Zero-width frames concatenate by adding their retained row counts.
//
// Errors: ErrSchemaMismatch, ErrRowCount.
func (f Frame) Concat(others ...Frame) (Frame, error) {
	if len(others) == 0 {
		return f, nil
	}
	total := f.Len()
	for frameIndex, other := range others {
		if other.Width() != f.Width() {
			return Frame{}, fmt.Errorf("%w: frame %d has %d columns, want %d", ErrSchemaMismatch, frameIndex+1, other.Width(), f.Width())
		}
		for columnIndex, column := range f.columns {
			otherColumn := other.columns[columnIndex]
			if otherColumn.name != column.name || otherColumn.typeOf != column.typeOf {
				return Frame{}, fmt.Errorf("%w: frame %d column %d is %q:%v, want %q:%v", ErrSchemaMismatch, frameIndex+1, columnIndex, otherColumn.name, otherColumn.typeOf, column.name, column.typeOf)
			}
		}
		if other.Len() > math.MaxInt-total {
			return Frame{}, fmt.Errorf("%w: concatenated row count overflows int", ErrRowCount)
		}
		total += other.Len()
	}
	if f.Width() == 0 {
		return Frame{rowCount: total}, nil
	}

	columns := make([]column, f.Width())
	parts := make([]columnData, len(others))
	for columnIndex, base := range f.columns {
		for i, other := range others {
			part := other.columns[columnIndex]
			parts[i] = part.values
			base.nullable = base.nullable || part.nullable
		}
		base.values = base.values.concat(parts, total, base.nullable)
		columns[columnIndex] = base
	}
	return Frame{columns: columns, rowCount: total}, nil
}
