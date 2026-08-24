package record

import (
	"errors"
	"reflect"
	"runtime"
	"testing"

	"github.com/joeychilson/dataframe/series"
)

func TestFieldPresenceTransitions_ExtractAndInitializeNullableValues(t *testing.T) {
	type value struct {
		Plain    any
		Pointer  *string
		Optional series.Optional[int]
	}
	fields, err := Describe(reflect.TypeFor[value]())
	if err != nil {
		t.Fatal(err)
	}

	source := value{Optional: series.Optional[int]{Value: 42}}
	sourceValue := reflect.ValueOf(source)
	plain, present := fields[0].Extract(sourceValue)
	if !present || plain.Kind() != reflect.Interface || !plain.IsNil() {
		t.Fatalf("nil interface extraction = (%v, %t), want present nil interface", plain, present)
	}
	if _, pointerPresent := fields[1].Extract(sourceValue); pointerPresent {
		t.Fatal("nil pointer extracted as present")
	}
	if _, optionalPresent := fields[2].Extract(sourceValue); optionalPresent {
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

func TestDescribe_MapsAndCachesRecordFields(t *testing.T) {
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
	typeOf := reflect.TypeOf(value{hidden: true})
	fields, err := Describe(typeOf)
	if err != nil {
		t.Fatal(err)
	}
	if len(fields) != 4 {
		t.Fatalf("field count = %d", len(fields))
	}
	cached, err := Describe(typeOf)
	if err != nil {
		t.Fatal(err)
	}
	if &cached[0] != &fields[0] {
		t.Fatal("Describe did not reuse successful metadata")
	}
	want := []struct {
		name      string
		valueType reflect.Type
		nullable  bool
	}{
		{name: "active", valueType: reflect.TypeFor[bool]()},
		{name: "id", valueType: reflect.TypeFor[int]()},
		{name: "name", valueType: reflect.TypeFor[string](), nullable: true},
		{name: "score", valueType: reflect.TypeFor[float64](), nullable: true},
	}
	for i, field := range fields {
		if field.Name != want[i].name || field.ValueType != want[i].valueType || field.Nullable() != want[i].nullable {
			t.Fatalf("field %d = %#v", i, field)
		}
	}
}

func TestDescribe_RejectsInvalidAndDuplicateFields(t *testing.T) {
	if _, err := Describe(reflect.TypeFor[*struct{}]()); !errors.Is(err, ErrInvalidRecord) {
		t.Fatalf("invalid type error = %v", err)
	}
	type duplicate struct {
		Left  int `df:"value"`
		Right int `df:"value"`
	}
	if _, err := Describe(reflect.TypeFor[duplicate]()); !errors.Is(err, ErrColumnConflict) {
		t.Fatalf("duplicate error = %v", err)
	}
	if _, err := Describe(reflect.TypeFor[duplicate]()); !errors.Is(err, ErrColumnConflict) {
		t.Fatalf("repeated duplicate error = %v", err)
	}
}

func BenchmarkDescribe(b *testing.B) {
	type Embedded struct {
		Active bool `df:"active"`
	}
	type value struct {
		Embedded
		ID    int                      `df:"id"`
		Name  *string                  `df:"name"`
		Score series.Optional[float64] `df:"score"`
	}
	typeOf := reflect.TypeFor[value]()
	b.Run("Uncached", func(b *testing.B) {
		b.ReportAllocs()
		var fields []Field
		for b.Loop() {
			var err error
			fields, err = describe(typeOf)
			if err != nil {
				b.Fatal(err)
			}
		}
		runtime.KeepAlive(fields)
	})

	b.Run("Cached", func(b *testing.B) {
		b.ReportAllocs()
		var fields []Field
		for b.Loop() {
			var err error
			fields, err = Describe(typeOf)
			if err != nil {
				b.Fatal(err)
			}
		}
		runtime.KeepAlive(fields)
	})
}
