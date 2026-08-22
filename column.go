package dataframe

import (
	"fmt"
	"iter"
	"reflect"

	"github.com/joeychilson/dataframe/internal/bitmap"
	"github.com/joeychilson/dataframe/mask"
	"github.com/joeychilson/dataframe/series"
)

// ColumnView is a read-only, dynamically typed view of one Frame column. It is
// intended for schema-driven code such as encoders, formatters, and adapters.
// Typed application code should use [Frame.Column] instead.
//
// The zero value is an unnamed, zero-length view whose Type is nil.
type ColumnView struct {
	name     string
	values   ColumnSpec
	typeOf   reflect.Type
	nullable bool
	length   int
}

// Columns iterates read-only column views in schema order.
func (f Frame) Columns() iter.Seq[ColumnView] {
	return func(yield func(ColumnView) bool) {
		for _, column := range f.columns {
			view := ColumnView{
				name:     column.columnName(),
				values:   column,
				typeOf:   column.columnType(),
				nullable: column.columnNullable(),
				length:   column.columnLen(),
			}
			if !yield(view) {
				return
			}
		}
	}
}

// Name returns the column name. It returns the empty string for the zero view;
// Frames never contain empty column names.
func (c ColumnView) Name() string {
	return c.name
}

// Len returns the column's row count.
func (c ColumnView) Len() int {
	return c.length
}

// Type returns the exact Go element type, or nil for the zero view.
func (c ColumnView) Type() reflect.Type {
	return c.typeOf
}

// Nullable reports whether the column's schema permits null cells.
func (c ColumnView) Nullable() bool {
	return c.nullable
}

// At returns row i and whether it is present. A null cell returns nil, false.
// It panics when i is out of range.
func (c ColumnView) At(i int) (any, bool) {
	if i < 0 || i >= c.length {
		panic("dataframe: ColumnView.At: index out of range")
	}
	return c.values.columnAt(i)
}

func columnFromSlice(name string, values reflect.Value, validity []bool) ColumnSpec {
	switch typed := values.Interface().(type) {
	case []bool:
		return typedColumnFromSlice(name, typed, validity)
	case []string:
		return typedColumnFromSlice(name, typed, validity)
	case []int:
		return typedColumnFromSlice(name, typed, validity)
	case []int8:
		return typedColumnFromSlice(name, typed, validity)
	case []int16:
		return typedColumnFromSlice(name, typed, validity)
	case []int32:
		return typedColumnFromSlice(name, typed, validity)
	case []int64:
		return typedColumnFromSlice(name, typed, validity)
	case []uint:
		return typedColumnFromSlice(name, typed, validity)
	case []uint8:
		return typedColumnFromSlice(name, typed, validity)
	case []uint16:
		return typedColumnFromSlice(name, typed, validity)
	case []uint32:
		return typedColumnFromSlice(name, typed, validity)
	case []uint64:
		return typedColumnFromSlice(name, typed, validity)
	case []uintptr:
		return typedColumnFromSlice(name, typed, validity)
	case []float32:
		return typedColumnFromSlice(name, typed, validity)
	case []float64:
		return typedColumnFromSlice(name, typed, validity)
	case []complex64:
		return typedColumnFromSlice(name, typed, validity)
	case []complex128:
		return typedColumnFromSlice(name, typed, validity)
	default:
		column := reflectColumnSpec{
			name:   name,
			typeOf: values.Type().Elem(),
			values: values,
		}
		if validity != nil {
			column.validity = bitmap.FromBools(validity)
		}
		return column
	}
}

func typedColumnFromSlice[T any](name string, values []T, validity []bool) ColumnSpec {
	if validity == nil {
		return Column(name, values)
	}
	typed, err := series.NewNullable(values, validity)
	if err != nil {
		panic(err)
	}
	return ColumnFromSeries(name, typed)
}

type reflectColumnSpec struct {
	name     string
	typeOf   reflect.Type
	values   reflect.Value
	validity bitmap.Bitmap
}

func (c reflectColumnSpec) dataframeColumnSpec() {}

func (c reflectColumnSpec) columnName() string {
	return c.name
}

func (c reflectColumnSpec) columnType() reflect.Type {
	return c.typeOf
}

func (c reflectColumnSpec) columnNullable() bool {
	return c.validity.Initialized()
}

func (c reflectColumnSpec) columnLen() int {
	return c.values.Len()
}

func (c reflectColumnSpec) columnAt(i int) (any, bool) {
	value := c.values.Index(i)
	if c.validity.Initialized() && !c.validity.At(i) {
		return nil, false
	}
	return value.Interface(), true
}

func (c reflectColumnSpec) columnRename(name string) ColumnSpec {
	c.name = name
	return c
}

func (c reflectColumnSpec) columnTake(rows []int) ColumnSpec {
	values := reflect.MakeSlice(reflect.SliceOf(c.typeOf), len(rows), len(rows))
	var validity bitmap.Bitmap
	if c.validity.Initialized() {
		validity = bitmap.New(len(rows))
	}
	for i, row := range rows {
		values.Index(i).Set(c.values.Index(row))
		if validity.Initialized() && c.validity.At(row) {
			validity.Set(i, true)
		}
	}
	return reflectColumnSpec{name: c.name, typeOf: c.typeOf, values: values, validity: validity}
}

func (c reflectColumnSpec) columnTakeNullable(rows series.Series[int]) ColumnSpec {
	values := reflect.MakeSlice(reflect.SliceOf(c.typeOf), rows.Len(), rows.Len())
	validity := bitmap.New(rows.Len())
	for i := 0; i < rows.Len(); i++ {
		row, present := rows.At(i)
		if !present {
			continue
		}
		values.Index(i).Set(c.values.Index(row))
		if !c.validity.Initialized() || c.validity.At(row) {
			validity.Set(i, true)
		}
	}
	return reflectColumnSpec{name: c.name, typeOf: c.typeOf, values: values, validity: validity}
}

func (c reflectColumnSpec) columnSlice(start, end int) ColumnSpec {
	result := reflectColumnSpec{
		name:   c.name,
		typeOf: c.typeOf,
		values: c.values.Slice3(start, end, end),
	}
	if c.validity.Initialized() {
		result.validity = c.validity.Slice(start, end)
	}
	return result
}

func (c reflectColumnSpec) columnFilter(selection mask.Mask) ColumnSpec {
	values := reflect.MakeSlice(reflect.SliceOf(c.typeOf), selection.Count(), selection.Count())
	var validity bitmap.Bitmap
	if c.validity.Initialized() {
		validity = bitmap.New(values.Len())
	}
	i := 0
	for row := range selection.Rows() {
		values.Index(i).Set(c.values.Index(row))
		if validity.Initialized() && c.validity.At(row) {
			validity.Set(i, true)
		}
		i++
	}
	return reflectColumnSpec{name: c.name, typeOf: c.typeOf, values: values, validity: validity}
}

func (c reflectColumnSpec) columnConcat(others []ColumnSpec) (ColumnSpec, error) {
	maxInt := int(^uint(0) >> 1)
	total := c.columnLen()
	nullable := c.columnNullable()
	for _, other := range others {
		if other.columnType() != c.typeOf {
			return nil, fmt.Errorf("%w: column %q type %v does not match %v", ErrSchemaMismatch, c.name, other.columnType(), c.typeOf)
		}
		if other.columnLen() > maxInt-total {
			return nil, fmt.Errorf("%w: concatenated column %q length overflows int", ErrRowCount, c.name)
		}
		total += other.columnLen()
		nullable = nullable || other.columnNullable()
	}

	values := reflect.MakeSlice(reflect.SliceOf(c.typeOf), total, total)
	var validity bitmap.Bitmap
	if nullable {
		validity = bitmap.Filled(total)
	}
	all := make([]ColumnSpec, 0, len(others)+1)
	all = append(all, c)
	all = append(all, others...)
	offset := 0
	for _, column := range all {
		for row := 0; row < column.columnLen(); row++ {
			value, present := column.columnAt(row)
			if present && value != nil {
				values.Index(offset + row).Set(reflect.ValueOf(value))
			}
			if validity.Initialized() && !present {
				validity.Set(offset+row, false)
			}
		}
		offset += column.columnLen()
	}
	return reflectColumnSpec{name: c.name, typeOf: c.typeOf, values: values, validity: validity}, nil
}

func typedSeriesFromColumn[T any](column ColumnSpec) (series.Series[T], error) {
	if typed, ok := column.(columnSpec[T]); ok {
		return typed.values, nil
	}
	values := make([]T, column.columnLen())
	var validity []bool
	if column.columnNullable() {
		validity = make([]bool, len(values))
	}
	for i := range values {
		value, present := column.columnAt(i)
		if present {
			if value != nil {
				converted, ok := value.(T)
				if !ok {
					return series.Series[T]{}, fmt.Errorf("%w: column %q contains %T, want %v", ErrColumnType, column.columnName(), value, reflect.TypeFor[T]())
				}
				values[i] = converted
			}
			if validity != nil {
				validity[i] = true
			}
		}
	}
	if validity == nil {
		return series.New(values), nil
	}
	result, err := series.NewNullable(values, validity)
	if err != nil {
		panic(err)
	}
	return result, nil
}
