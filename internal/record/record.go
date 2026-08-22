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

// Kind describes how a record field represents presence.
type Kind uint8

const (
	// Value is an ordinary non-null field.
	Value Kind = iota
	// Pointer is a nullable pointer field whose element is the column value.
	Pointer
	// Optional is a nullable series.Optional field.
	Optional
)

// Field describes one mapped record field.
type Field struct {
	// Name is the mapped column name.
	Name string
	// Index is the reflection index path from the record root.
	Index []int
	// ValueType is the column type after removing the nullable wrapper.
	ValueType reflect.Type
	// Kind describes the field's representation of presence.
	Kind Kind
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

			described := Field{Name: name, Index: index, ValueType: field.Type}
			switch {
			case field.Type.Kind() == reflect.Pointer:
				described.Kind = Pointer
				described.ValueType = field.Type.Elem()
			case isOptional(field.Type):
				described.Kind = Optional
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
