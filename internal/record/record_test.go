package record

import (
	"errors"
	"reflect"
	"testing"

	"github.com/joeychilson/dataframe/series"
)

func TestDescribe(t *testing.T) {
	type Embedded struct {
		Active bool `df:"active"`
	}
	type value struct {
		Embedded
		ID      int                      `df:"id"`
		Name    *string                  `df:"name"`
		Score   series.Optional[float64] `df:"score"`
		Ignored string                   `df:"-"`
		hidden  bool
	}
	invalid := errors.New("invalid record")
	invalidName := errors.New("invalid name")
	conflict := errors.New("conflict")
	fields, err := Describe(reflect.TypeFor[value](), invalid, invalidName, conflict)
	if err != nil {
		t.Fatal(err)
	}
	if len(fields) != 4 {
		t.Fatalf("field count = %d", len(fields))
	}
	want := []struct {
		name      string
		valueType reflect.Type
		kind      Kind
	}{
		{name: "active", valueType: reflect.TypeFor[bool](), kind: Value},
		{name: "id", valueType: reflect.TypeFor[int](), kind: Value},
		{name: "name", valueType: reflect.TypeFor[string](), kind: Pointer},
		{name: "score", valueType: reflect.TypeFor[float64](), kind: Optional},
	}
	for i, field := range fields {
		if field.Name != want[i].name || field.ValueType != want[i].valueType || field.Kind != want[i].kind {
			t.Fatalf("field %d = %#v", i, field)
		}
	}
}

func TestDescribeErrors(t *testing.T) {
	invalid := errors.New("invalid record")
	invalidName := errors.New("invalid name")
	conflict := errors.New("conflict")
	if _, err := Describe(reflect.TypeFor[*struct{}](), invalid, invalidName, conflict); !errors.Is(err, invalid) {
		t.Fatalf("invalid type error = %v", err)
	}
	type duplicate struct {
		Left  int `df:"value"`
		Right int `df:"value"`
	}
	if _, err := Describe(reflect.TypeFor[duplicate](), invalid, invalidName, conflict); !errors.Is(err, conflict) {
		t.Fatalf("duplicate error = %v", err)
	}
}
