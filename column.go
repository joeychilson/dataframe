package dataframe

import (
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
	length int
}

// Columns iterates read-only column views in schema order.
func (f Frame) Columns() iter.Seq[ColumnView] {
	length := f.Len()
	return func(yield func(ColumnView) bool) {
		for i := range f.columns {
			if !yield(ColumnView{column: f.columns[i], length: length}) {
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
	return c.length
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

type column struct {
	name     string
	typeOf   reflect.Type
	nullable bool
	values   columnData
}

// columnData is the sealed storage sum. Built-in columns and columns created
// by New or With use typedData; non-built-in FromRecords columns use
// reflectData and are converted by typedSeriesFromData at Frame.Column.
type columnData interface {
	at(int) (any, bool)
	take([]int) columnData
	takeNullable(series.Series[int]) columnData
	slice(int, int) columnData
	filter(mask.Mask) columnData
	length() int
	concat([]columnData, int, bool) columnData
}

type columnSpec[T any] struct {
	name   string
	values series.Series[T]
}

type typedData[T any] struct {
	values series.Series[T]
}

func (c columnSpec[T]) dataframeColumnSpec() (column, int) {
	return typedColumn(c.name, c.values), c.values.Len()
}

func typedColumn[T any](name string, values series.Series[T]) column {
	return column{
		name:     name,
		typeOf:   reflect.TypeFor[T](),
		nullable: values.Nullable(),
		values:   typedData[T]{values: values},
	}
}

func (c typedData[T]) at(i int) (any, bool) {
	value, present := c.values.At(i)
	if !present {
		return nil, false
	}
	return value, true
}

func (c typedData[T]) take(rows []int) columnData {
	c.values = c.values.Take(rows)
	return c
}

func (c typedData[T]) takeNullable(rows series.Series[int]) columnData {
	c.values = c.values.TakeNullable(rows)
	return c
}

func (c typedData[T]) slice(start, end int) columnData {
	c.values = c.values.Slice(start, end)
	return c
}

func (c typedData[T]) filter(selection mask.Mask) columnData {
	c.values = c.values.Filter(selection)
	return c
}

func (c typedData[T]) length() int {
	return c.values.Len()
}

func (c typedData[T]) concat(others []columnData, _ int, _ bool) columnData {
	parts := make([]series.Series[T], len(others))
	for i, other := range others {
		parts[i] = typedSeriesFromData[T](other)
	}
	c.values = c.values.Concat(parts...)
	return c
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

func (c reflectData) length() int {
	return c.values.Len()
}

func (c reflectData) concat(others []columnData, total int, nullable bool) columnData {
	values := reflect.MakeSlice(c.values.Type(), total, total)
	var validity bitmap.Bitmap
	if nullable {
		validity = bitmap.Filled(total)
	}
	reflect.Copy(values, c.values)
	if c.validity.Initialized() {
		validity.Copy(0, c.validity)
	}
	offset := c.values.Len()
	for _, other := range others {
		length := other.length()
		if reflected, ok := other.(reflectData); ok {
			reflect.Copy(values.Slice(offset, offset+length), reflected.values)
			if reflected.validity.Initialized() {
				validity.Copy(offset, reflected.validity)
			}
		} else {
			for row := range length {
				value, present := other.at(row)
				if present && value != nil {
					values.Index(offset + row).Set(reflect.ValueOf(value))
				}
				if nullable && !present {
					validity.Set(offset+row, false)
				}
			}
		}
		offset += length
	}
	return reflectData{values: values, validity: validity}
}

func typedSeriesFromData[T any](data columnData) series.Series[T] {
	if typed, ok := data.(typedData[T]); ok {
		return typed.values
	}
	reflected := data.(reflectData)
	// Callers have already checked the exact element type.
	values := reflected.values.Interface().([]T)
	if !reflected.validity.Initialized() {
		return series.New(values)
	}
	result, err := series.NewNullable(values, reflected.validity.Bools())
	if err != nil {
		panic(err)
	}
	return result
}
