package dataframe

import (
	"fmt"
	"iter"
	"reflect"

	"github.com/joeychilson/dataframe/internal/record"
)

// Row is a read-only view of one heterogeneous Frame row. Column-oriented
// operations should be preferred for bulk work.
type Row struct {
	frame *Frame
	index int
}

// Rows iterates rows in order.
func (f Frame) Rows() iter.Seq2[int, Row] {
	return func(yield func(int, Row) bool) {
		for i := 0; i < f.Len(); i++ {
			if !yield(i, Row{frame: &f, index: i}) {
				return
			}
		}
	}
}

// Row returns row i. It panics when i is out of range.
func (f Frame) Row(i int) Row {
	if i < 0 || i >= f.Len() {
		panic("dataframe: Row: index out of range")
	}
	return Row{frame: &f, index: i}
}

// Get returns column, whether its cell is present, and any dynamic-schema
// error. Missing columns return ErrColumnNotFound and type mismatches return
// ErrColumnType. The zero Row returns ErrInvalidRow.
func (r Row) Get[T any](column string) (value T, present bool, err error) {
	if r.frame == nil {
		return value, false, fmt.Errorf("%w: zero value", ErrInvalidRow)
	}
	index := r.frame.columnIndex(column)
	if index < 0 {
		return value, false, fmt.Errorf("%w: %q", ErrColumnNotFound, column)
	}
	stored := r.frame.columns[index]
	want := reflect.TypeFor[T]()
	if stored.columnType() != want {
		return value, false, fmt.Errorf("%w: column %q has type %v, want %v", ErrColumnType, column, stored.columnType(), want)
	}
	if typed, ok := stored.(columnSpec[T]); ok {
		value, present = typed.values.At(r.index)
		return value, present, nil
	}
	dynamic, present := stored.columnAt(r.index)
	if !present || dynamic == nil {
		return value, present, nil
	}
	value, ok := dynamic.(T)
	if !ok {
		return value, false, fmt.Errorf("%w: column %q contains %T, want %v", ErrColumnType, column, dynamic, want)
	}
	return value, true, nil
}

// FromRecords builds a Frame from records of non-pointer struct type T. Empty
// input retains T's schema. Fields are mapped as follows:
//
//   - exported fields participate; `df:"-"` ignores a field;
//   - `df:"name"` sets a column name, otherwise the field name is used;
//   - anonymous embedded structs are flattened;
//   - pointer-to-U and series.Optional[U] fields create nullable U columns;
//   - other field types create exact, non-null columns of that Go type.
//
// Duplicate mapped names and a T that is not a non-pointer struct return an
// error.
func FromRecords[T any](records []T) (Frame, error) {
	typeOf := reflect.TypeFor[T]()
	fields, err := record.Describe(typeOf, ErrInvalidRecord, ErrInvalidName, ErrColumnConflict)
	if err != nil {
		return Frame{}, err
	}
	if len(fields) == 0 {
		return Frame{rowCount: len(records)}, nil
	}

	columns := make([]ColumnSpec, len(fields))
	recordValues := reflect.ValueOf(records)
	for i, field := range fields {
		values := reflect.MakeSlice(reflect.SliceOf(field.ValueType), len(records), len(records))
		var validity []bool
		if field.Kind != record.Value {
			validity = make([]bool, len(records))
		}
		for row := range records {
			value := recordValues.Index(row).FieldByIndex(field.Index)
			switch field.Kind {
			case record.Value:
				values.Index(row).Set(value)
			case record.Pointer:
				if !value.IsNil() {
					values.Index(row).Set(value.Elem())
					validity[row] = true
				}
			case record.Optional:
				if value.Field(1).Bool() {
					values.Index(row).Set(value.Field(0))
					validity[row] = true
				}
			}
		}
		columns[i] = columnFromSlice(field.Name, values, validity)
	}
	return New(columns...)
}

// Records materializes f as records of non-pointer struct type T. Extra frame
// columns are ignored. Every mapped field must find an exactly typed column;
// null in a plain field is an error.
func (f Frame) Records[T any]() ([]T, error) {
	typeOf := reflect.TypeFor[T]()
	fields, err := record.Describe(typeOf, ErrInvalidRecord, ErrInvalidName, ErrColumnConflict)
	if err != nil {
		return nil, err
	}
	columns := make([]ColumnSpec, len(fields))
	for i, field := range fields {
		index := f.columnIndex(field.Name)
		if index < 0 {
			return nil, fmt.Errorf("%w: %q", ErrColumnNotFound, field.Name)
		}
		column := f.columns[index]
		if column.columnType() != field.ValueType {
			return nil, fmt.Errorf("%w: column %q has type %v, want %v", ErrColumnType, field.Name, column.columnType(), field.ValueType)
		}
		columns[i] = column
	}

	records := make([]T, f.Len())
	for row := range records {
		recordValue := reflect.ValueOf(&records[row]).Elem()
		for i, field := range fields {
			value, present := columns[i].columnAt(row)
			destination := recordValue.FieldByIndex(field.Index)
			switch field.Kind {
			case record.Value:
				if !present {
					return nil, fmt.Errorf("%w: null in non-null field %s at row %d", ErrInvalidRecord, field.Name, row)
				}
				if value != nil {
					destination.Set(reflect.ValueOf(value))
				}
			case record.Pointer:
				if present {
					destination.Set(reflect.New(field.ValueType))
					if value != nil {
						destination.Elem().Set(reflect.ValueOf(value))
					}
				}
			case record.Optional:
				if present {
					if value != nil {
						destination.Field(0).Set(reflect.ValueOf(value))
					}
					destination.Field(1).SetBool(true)
				}
			}
		}
	}
	return records, nil
}
