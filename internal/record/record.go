// Package record describes the struct fields shared by dataframe record
// conversion and its adapters.
package record

import (
	"fmt"
	"reflect"
	"slices"
	"strings"

	"github.com/joeychilson/dataframe/series"
)

type kind uint8

const (
	valueKind kind = iota
	pointerKind
	optionalKind
)

// Field describes one mapped record field.
type Field struct {
	// Name is the mapped column name.
	Name string
	// ValueType is the column type after removing the nullable wrapper.
	ValueType reflect.Type
	index     []int
	kind      kind
}

// Nullable reports whether the field can represent an absent value.
func (f Field) Nullable() bool {
	return f.kind != valueKind
}

// Extract returns the field's unwrapped value and whether it is present.
// Ordinary value fields are always present, including nil interface values.
func (f Field) Extract(record reflect.Value) (reflect.Value, bool) {
	value := record.FieldByIndex(f.index)
	switch f.kind {
	case valueKind:
		return value, true
	case pointerKind:
		if value.IsNil() {
			return reflect.Value{}, false
		}
		return value.Elem(), true
	case optionalKind:
		if !value.Field(1).Bool() {
			return reflect.Value{}, false
		}
		return value.Field(0), true
	default:
		panic("record: invalid field kind")
	}
}

// Destination returns the addressable unwrapped destination for a present
// field value. Pointer storage is allocated and Optional validity is set.
func (f Field) Destination(record reflect.Value) reflect.Value {
	destination := record.FieldByIndex(f.index)
	switch f.kind {
	case valueKind:
		return destination
	case pointerKind:
		destination.Set(reflect.New(f.ValueType))
		return destination.Elem()
	case optionalKind:
		destination.Field(1).SetBool(true)
		return destination.Field(0)
	default:
		panic("record: invalid field kind")
	}
}

// Describe returns the mapped fields of a non-pointer struct type. The caller
// supplies sentinel errors so returned errors participate in its public error
// contract.
func Describe(typeOf reflect.Type, invalidRecord, invalidName, columnConflict error) ([]Field, error) {
	if typeOf == nil || typeOf.Kind() != reflect.Struct {
		return nil, fmt.Errorf("%w: got %v", invalidRecord, typeOf)
	}

	var fields []Field
	names := make(map[string]struct{})
	var walk func(reflect.Type, []int) error
	walk = func(current reflect.Type, prefix []int) error {
		for i := 0; i < current.NumField(); i++ {
			field := current.Field(i)
			if field.PkgPath != "" {
				continue
			}
			tag, tagged := field.Tag.Lookup("df")
			name, _, _ := strings.Cut(tag, ",")
			if tagged && name == "-" {
				continue
			}
			index := append(slices.Clone(prefix), i)
			if field.Anonymous && field.Type.Kind() == reflect.Struct && !tagged {
				if err := walk(field.Type, index); err != nil {
					return err
				}
				continue
			}
			if !tagged || name == "" {
				name = field.Name
			}
			if name == "" {
				return fmt.Errorf("%w: field %s", invalidName, field.Name)
			}
			if _, exists := names[name]; exists {
				return fmt.Errorf("%w: record field name %q", columnConflict, name)
			}

			described := Field{Name: name, index: index, ValueType: field.Type}
			switch {
			case field.Type.Kind() == reflect.Pointer:
				described.kind = pointerKind
				described.ValueType = field.Type.Elem()
			case isOptional(field.Type):
				described.kind = optionalKind
				described.ValueType = field.Type.Field(0).Type
			}
			names[name] = struct{}{}
			fields = append(fields, described)
		}
		return nil
	}
	if err := walk(typeOf, nil); err != nil {
		return nil, err
	}
	return fields, nil
}

func isOptional(typeOf reflect.Type) bool {
	return typeOf.Kind() == reflect.Struct &&
		typeOf.PkgPath() == reflect.TypeFor[series.Optional[struct{}]]().PkgPath() &&
		strings.HasPrefix(typeOf.Name(), "Optional[") &&
		typeOf.NumField() == 2 &&
		typeOf.Field(0).Name == "Value" &&
		typeOf.Field(1).Name == "Valid" &&
		typeOf.Field(1).Type == reflect.TypeFor[bool]()
}
