package dataframe

import (
	"fmt"
	"iter"
	"reflect"
	"slices"

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
		for i := range f.Len() {
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
func (r Row) Get[T any](name string) (value T, present bool, err error) {
	if r.frame == nil {
		return value, false, fmt.Errorf("%w: zero value", ErrInvalidRow)
	}
	index := slices.IndexFunc(r.frame.columns, func(stored column) bool {
		return stored.name == name
	})
	if index < 0 {
		return value, false, fmt.Errorf("%w: %q", ErrColumnNotFound, name)
	}
	stored := r.frame.columns[index]
	want := reflect.TypeFor[T]()
	if stored.typeOf != want {
		return value, false, fmt.Errorf("%w: column %q has type %v, want %v", ErrColumnType, name, stored.typeOf, want)
	}
	if typed, ok := stored.values.(typedData[T]); ok {
		value, present = typed.values.At(r.index)
		return value, present, nil
	}
	dynamic, present := stored.values.at(r.index)
	if !present || dynamic == nil {
		return value, present, nil
	}
	return dynamic.(T), true, nil
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
	fields, err := record.Describe(reflect.TypeFor[T]())
	if err != nil {
		return Frame{}, err
	}
	if len(fields) == 0 {
		return Frame{rowCount: len(records)}, nil
	}

	columns := make([]column, len(fields))
	recordValues := reflect.ValueOf(records)
	for i, field := range fields {
		values := reflect.MakeSlice(reflect.SliceOf(field.ValueType), len(records), len(records))
		var validity []bool
		if field.Nullable() {
			validity = make([]bool, len(records))
		}
		for row := range records {
			value, present := field.Extract(recordValues.Index(row))
			if present {
				values.Index(row).Set(value)
				if validity != nil {
					validity[row] = true
				}
			}
		}
		columns[i] = columnFromSlice(field.Name, values, validity)
	}
	return Frame{columns: columns, rowCount: len(records)}, nil
}

// Records materializes f as records of non-pointer struct type T. Extra frame
// columns are ignored. Every mapped field must find an exactly typed column;
// null in a plain field is an error.
func (f Frame) Records[T any]() ([]T, error) {
	fields, err := record.Describe(reflect.TypeFor[T]())
	if err != nil {
		return nil, err
	}
	columns := make([]column, len(fields))
	for i, field := range fields {
		index := slices.IndexFunc(f.columns, func(column column) bool {
			return column.name == field.Name
		})
		if index < 0 {
			return nil, fmt.Errorf("%w: %q", ErrColumnNotFound, field.Name)
		}
		stored := f.columns[index]
		if stored.typeOf != field.ValueType {
			return nil, fmt.Errorf("%w: column %q has type %v, want %v", ErrColumnType, field.Name, stored.typeOf, field.ValueType)
		}
		columns[i] = stored
	}

	records := make([]T, f.Len())
	for row := range records {
		recordValue := reflect.ValueOf(&records[row]).Elem()
		for i, field := range fields {
			value, present := columns[i].values.at(row)
			if !present {
				if !field.Nullable() {
					return nil, fmt.Errorf("%w: null in non-null field %s at row %d", ErrInvalidRecord, field.Name, row)
				}
				continue
			}
			destination := field.Destination(recordValue)
			if value != nil {
				destination.Set(reflect.ValueOf(value))
			}
		}
	}
	return records, nil
}
