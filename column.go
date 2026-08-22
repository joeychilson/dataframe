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
	column column
}

// Columns iterates read-only column views in schema order.
func (f Frame) Columns() iter.Seq[ColumnView] {
	return func(yield func(ColumnView) bool) {
		for _, column := range f.columns {
			if !yield(ColumnView{column: column}) {
				return
			}
		}
	}
}

// Name returns the column name. It returns the empty string for the zero view;
// Frames never contain empty column names.
func (c ColumnView) Name() string {
	return c.column.name
}

// Len returns the column's row count.
func (c ColumnView) Len() int {
	return c.column.length
}

// Type returns the exact Go element type, or nil for the zero view.
func (c ColumnView) Type() reflect.Type {
	return c.column.typeOf
}

// Nullable reports whether the column's schema permits null cells.
func (c ColumnView) Nullable() bool {
	return c.column.nullable
}

// At returns row i and whether it is present. A null cell returns nil, false.
// It panics when i is out of range.
func (c ColumnView) At(i int) (any, bool) {
	return c.column.values.at(i)
}

func columnFromSlice(name string, values reflect.Value, validity []bool) column {
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
		data := reflectData{values: values}
		if validity != nil {
			data.validity = bitmap.FromBools(validity)
		}
		return column{
			name:     name,
			typeOf:   values.Type().Elem(),
			nullable: data.validity.Initialized(),
			length:   values.Len(),
			values:   data,
		}
	}
}

func typedColumnFromSlice[T any](name string, values []T, validity []bool) column {
	if validity == nil {
		return typedColumn(name, series.New(values))
	}
	typed, err := series.NewNullable(values, validity)
	if err != nil {
		panic(err)
	}
	return typedColumn(name, typed)
}

type reflectData struct {
	values   reflect.Value
	validity bitmap.Bitmap
}

func (c reflectData) at(i int) (any, bool) {
	value := c.values.Index(i)
	if c.validity.Initialized() && !c.validity.At(i) {
		return nil, false
	}
	return value.Interface(), true
}

func (c reflectData) take(rows []int) columnData {
	values := reflect.MakeSlice(c.values.Type(), len(rows), len(rows))
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
	return reflectData{values: values, validity: validity}
}

func (c reflectData) takeNullable(rows series.Series[int]) columnData {
	values := reflect.MakeSlice(c.values.Type(), rows.Len(), rows.Len())
	validity := bitmap.New(rows.Len())
	for i, row := range rows.Present() {
		values.Index(i).Set(c.values.Index(row))
		if !c.validity.Initialized() || c.validity.At(row) {
			validity.Set(i, true)
		}
	}
	return reflectData{values: values, validity: validity}
}

func (c reflectData) slice(start, end int) columnData {
	result := reflectData{values: c.values.Slice3(start, end, end)}
	if c.validity.Initialized() {
		result.validity = c.validity.Slice(start, end)
	}
	return result
}

func (c reflectData) filter(selection mask.Mask) columnData {
	values := reflect.MakeSlice(c.values.Type(), selection.Count(), selection.Count())
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
	return reflectData{values: values, validity: validity}
}

func (c reflectData) concat(base column, others []column) (columnData, error) {
	maxInt := int(^uint(0) >> 1)
	total := base.length
	nullable := base.nullable
	for _, other := range others {
		if other.typeOf != base.typeOf {
			return nil, fmt.Errorf("%w: column %q type %v does not match %v", ErrSchemaMismatch, base.name, other.typeOf, base.typeOf)
		}
		if other.length > maxInt-total {
			return nil, fmt.Errorf("%w: concatenated column %q length overflows int", ErrRowCount, base.name)
		}
		total += other.length
		nullable = nullable || other.nullable
	}

	values := reflect.MakeSlice(c.values.Type(), total, total)
	var validity bitmap.Bitmap
	if nullable {
		validity = bitmap.Filled(total)
	}
	all := make([]column, 0, len(others)+1)
	all = append(all, base)
	all = append(all, others...)
	offset := 0
	for _, column := range all {
		for row := range column.length {
			value, present := column.values.at(row)
			if present && value != nil {
				values.Index(offset + row).Set(reflect.ValueOf(value))
			}
			if validity.Initialized() && !present {
				validity.Set(offset+row, false)
			}
		}
		offset += column.length
	}
	return reflectData{values: values, validity: validity}, nil
}

func typedSeriesFromColumn[T any](column column) series.Series[T] {
	if typed, ok := column.values.(typedData[T]); ok {
		return typed.values
	}
	values := make([]T, column.length)
	var validity []bool
	if column.nullable {
		validity = make([]bool, len(values))
	}
	for i := range values {
		value, present := column.values.at(i)
		if present {
			if value != nil {
				values[i] = value.(T)
			}
			if validity != nil {
				validity[i] = true
			}
		}
	}
	if validity == nil {
		return series.New(values)
	}
	result, err := series.NewNullable(values, validity)
	if err != nil {
		panic(err)
	}
	return result
}
