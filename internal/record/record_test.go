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
		kind      kind
	}{
		{name: "active", valueType: reflect.TypeFor[bool](), kind: valueKind},
		{name: "id", valueType: reflect.TypeFor[int](), kind: valueKind},
		{name: "name", valueType: reflect.TypeFor[string](), kind: pointerKind},
		{name: "score", valueType: reflect.TypeFor[float64](), kind: optionalKind},
	}
	for i, field := range fields {
		if field.Name != want[i].name || field.ValueType != want[i].valueType || field.kind != want[i].kind {
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

func TestFieldPresenceTransitions(t *testing.T) {
	type value struct {
		Plain    any
		Pointer  *string
		Optional series.Optional[int]
	}
	fields, err := Describe(
		reflect.TypeFor[value](),
		errors.New("invalid record"),
		errors.New("invalid name"),
		errors.New("conflict"),
	)
	if err != nil {
		t.Fatal(err)
	}

	source := value{Optional: series.Optional[int]{Value: 42}}
	sourceValue := reflect.ValueOf(source)
	plain, present := fields[0].Extract(sourceValue)
	if !present || plain.Kind() != reflect.Interface || !plain.IsNil() {
		t.Fatalf("nil interface extraction = (%v, %t), want present nil interface", plain, present)
	}
	if _, present := fields[1].Extract(sourceValue); present {
		t.Fatal("nil pointer extracted as present")
	}
	if _, present := fields[2].Extract(sourceValue); present {
		t.Fatal("invalid Optional payload extracted as present")
	}

	destination := value{}
	destinationValue := reflect.ValueOf(&destination).Elem()
	plainDestination := fields[0].Destination(destinationValue)
	pointerDestination := fields[1].Destination(destinationValue)
	optionalDestination := fields[2].Destination(destinationValue)
	if !plainDestination.CanAddr() || !pointerDestination.CanAddr() || !optionalDestination.CanAddr() {
		t.Fatal("Destination returned an unaddressable value")
	}
	plainDestination.Set(reflect.ValueOf("plain"))
	pointerDestination.SetString("pointer")
	optionalDestination.SetInt(7)
	if destination.Plain != "plain" || destination.Pointer == nil || *destination.Pointer != "pointer" || destination.Optional != series.Some(7) {
		t.Fatalf("destination transitions = %#v", destination)
	}
}
