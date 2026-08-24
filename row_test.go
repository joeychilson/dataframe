package dataframe

import (
	"errors"
	"reflect"
	"runtime"
	"slices"
	"testing"

	"github.com/joeychilson/dataframe/internal/bitmap"
	"github.com/joeychilson/dataframe/mask"
	"github.com/joeychilson/dataframe/series"
)

func TestRowsAndRowGet_ExposeRowsInOrder(t *testing.T) {
	names := series.FromOptionals([]series.Optional[string]{series.Some("a"), series.None[string]()})
	frame, err := New(Column("id", []int{1, 2}), ColumnFromSeries("name", names))
	if err != nil {
		t.Fatal(err)
	}

	var ids []int
	for index, row := range frame.Rows() {
		id, present, getErr := row.Get[int]("id")
		if getErr != nil || !present {
			t.Fatalf("row %d id = %d, %v, %v", index, id, present, getErr)
		}
		ids = append(ids, id)
	}
	if !slices.Equal(ids, []int{1, 2}) {
		t.Fatalf("row ids = %v", ids)
	}
	if value, present, getErr := frame.Row(1).Get[string]("name"); getErr != nil || present || value != "" {
		t.Fatalf("null name = %q, %v, %v", value, present, getErr)
	}
	if _, _, getErr := frame.Row(0).Get[string]("id"); !errors.Is(getErr, ErrColumnType) {
		t.Fatalf("type error = %v", getErr)
	}
	if _, _, getErr := frame.Row(0).Get[int]("missing"); !errors.Is(getErr, ErrColumnNotFound) {
		t.Fatalf("missing error = %v", getErr)
	}
	func() {
		defer func() {
			if recover() == nil {
				t.Fatal("Row did not panic")
			}
		}()
		frame.Row(2)
	}()

	count := 0
	for range frame.Rows() {
		count++
		break
	}
	if count != 1 {
		t.Fatalf("early-stop count = %d", count)
	}
}

func TestZeroRowGetReturnsError(t *testing.T) {
	var row Row
	value, present, err := row.Get[int]("id")
	if value != 0 || present || !errors.Is(err, ErrInvalidRow) {
		t.Fatalf("zero Row.Get() = (%d, %t, %v), want (0, false, ErrInvalidRow)", value, present, err)
	}
}

func TestZeroWidthRowGetReturnsColumnError(t *testing.T) {
	frame, err := FromRecords([]struct{}{{}})
	if err != nil {
		t.Fatal(err)
	}
	if _, _, getErr := frame.Row(0).Get[int]("missing"); !errors.Is(getErr, ErrColumnNotFound) {
		t.Fatalf("zero-width Row.Get() error = %v, want ErrColumnNotFound", getErr)
	}
}

func TestFromRecordsAndRecords_RoundTripMappedFields(t *testing.T) {
	type Metadata struct {
		Active bool `df:"active"`
	}
	type record struct {
		Metadata
		ID      int                      `df:"id"`
		Name    *string                  `df:"name"`
		Score   series.Optional[float64] `df:"score"`
		Ignored string                   `df:"-"`
		hidden  int
	}
	a := "a"
	input := []record{
		{Active: true, ID: 1, Name: &a, Score: series.Some(1.5), Ignored: "x", hidden: 10},
		{ID: 2, Score: series.None[float64](), Ignored: "y", hidden: 20},
	}

	frame, err := FromRecords(input)
	if err != nil {
		t.Fatal(err)
	}
	wantSchema := []Field{
		{Name: "active", Type: reflect.TypeFor[bool]()},
		{Name: "id", Type: reflect.TypeFor[int]()},
		{Name: "name", Type: reflect.TypeFor[string](), Nullable: true},
		{Name: "score", Type: reflect.TypeFor[float64](), Nullable: true},
	}
	if got := frame.Schema(); !reflect.DeepEqual(got, wantSchema) {
		t.Fatalf("schema = %#v, want %#v", got, wantSchema)
	}

	got, err := frame.Records[record]()
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 || got[0].Name == nil || *got[0].Name != "a" || got[1].Name != nil || !got[0].Score.Valid || got[1].Score.Valid {
		t.Fatalf("round trip = %#v", got)
	}

	empty, err := FromRecords([]record{})
	if err != nil {
		t.Fatal(err)
	}
	if empty.Len() != 0 || !reflect.DeepEqual(empty.Schema(), wantSchema) {
		t.Fatalf("empty records = len %d schema %#v", empty.Len(), empty.Schema())
	}
}

func TestRecordRoundTripPreservesPresentNil(t *testing.T) {
	type record struct {
		Value    any
		Optional series.Optional[any]
	}
	frame, err := FromRecords([]record{{Optional: series.Some[any](nil)}})
	if err != nil {
		t.Fatal(err)
	}
	frame, err = frame.Concat(frame)
	if err != nil {
		t.Fatal(err)
	}
	records, err := frame.Records[record]()
	if err != nil {
		t.Fatal(err)
	}
	if len(records) != 2 {
		t.Fatalf("round trip = %#v", records)
	}
	for _, record := range records {
		if record.Value != nil || !record.Optional.Valid || record.Optional.Value != nil {
			t.Fatalf("round trip = %#v", records)
		}
	}
}

func TestFromRecordsUsesTypedStorageForBuiltins(t *testing.T) {
	type recordID int
	type record struct {
		Bool       bool
		String     string
		Int        int
		Int8       int8
		Int16      int16
		Int32      int32
		Int64      int64
		Uint       uint
		Uint8      uint8
		Uint16     uint16
		Uint32     uint32
		Uint64     uint64
		Uintptr    uintptr
		Float32    float32
		Float64    float64
		Complex64  complex64
		Complex128 complex128
		Pointer    *int
		Optional   series.Optional[string]
		Defined    recordID
		Slice      []int
	}
	value := 1
	frame, err := FromRecords([]record{{Pointer: &value, Optional: series.Some("value")}})
	if err != nil {
		t.Fatal(err)
	}
	assertTypedRecordColumn[bool](t, frame, "Bool")
	assertTypedRecordColumn[string](t, frame, "String")
	assertTypedRecordColumn[int](t, frame, "Int")
	assertTypedRecordColumn[int8](t, frame, "Int8")
	assertTypedRecordColumn[int16](t, frame, "Int16")
	assertTypedRecordColumn[int32](t, frame, "Int32")
	assertTypedRecordColumn[int64](t, frame, "Int64")
	assertTypedRecordColumn[uint](t, frame, "Uint")
	assertTypedRecordColumn[uint8](t, frame, "Uint8")
	assertTypedRecordColumn[uint16](t, frame, "Uint16")
	assertTypedRecordColumn[uint32](t, frame, "Uint32")
	assertTypedRecordColumn[uint64](t, frame, "Uint64")
	assertTypedRecordColumn[uintptr](t, frame, "Uintptr")
	assertTypedRecordColumn[float32](t, frame, "Float32")
	assertTypedRecordColumn[float64](t, frame, "Float64")
	assertTypedRecordColumn[complex64](t, frame, "Complex64")
	assertTypedRecordColumn[complex128](t, frame, "Complex128")
	assertTypedRecordColumn[int](t, frame, "Pointer")
	assertTypedRecordColumn[string](t, frame, "Optional")
	for _, name := range []string{"Pointer", "Optional"} {
		index := slices.IndexFunc(frame.columns, func(column column) bool {
			return column.name == name
		})
		if !frame.columns[index].nullable {
			t.Fatalf("column %q is not nullable", name)
		}
	}
	for _, name := range []string{"Defined", "Slice"} {
		index := slices.IndexFunc(frame.columns, func(column column) bool {
			return column.name == name
		})
		if _, ok := frame.columns[index].values.(reflectData); !ok {
			t.Fatalf("column %q storage is %T, want reflectData", name, frame.columns[index].values)
		}
	}
}

func TestFrameColumn_RetrievesRecordBackedValues(t *testing.T) {
	type value int
	type record struct {
		Value    value
		Optional *value
	}
	one, two := value(1), value(2)
	frame, err := FromRecords([]record{
		{Value: one, Optional: &two},
		{Value: two},
	})
	if err != nil {
		t.Fatal(err)
	}

	values, err := frame.Column[value]("Value")
	if err != nil {
		t.Fatal(err)
	}
	if got, want := values.Values(), []value{one, two}; !slices.Equal(got, want) {
		t.Fatalf("values = %v, want %v", got, want)
	}
	optional, err := frame.Column[value]("Optional")
	if err != nil {
		t.Fatal(err)
	}
	if got, want := optional.Optionals(), []series.Optional[value]{series.Some(two), series.None[value]()}; !slices.Equal(got, want) {
		t.Fatalf("optional values = %v, want %v", got, want)
	}
}

func TestRecordConversion_RejectsInvalidSchemasAndNulls(t *testing.T) {
	if _, err := FromRecords([]*struct{}{}); !errors.Is(err, ErrInvalidRecord) {
		t.Fatalf("pointer record error = %v", err)
	}
	type duplicate struct {
		A int `df:"same"`
		B int `df:"same"`
	}
	if _, err := FromRecords([]duplicate{}); !errors.Is(err, ErrColumnConflict) {
		t.Fatalf("duplicate field error = %v", err)
	}

	nullable := series.FromOptionals([]series.Optional[int]{series.None[int]()})
	frame, err := New(ColumnFromSeries("Value", nullable))
	if err != nil {
		t.Fatal(err)
	}
	if _, recordsErr := frame.Records[struct{ Value int }](); !errors.Is(recordsErr, ErrInvalidRecord) {
		t.Fatalf("null scalar error = %v", recordsErr)
	}
	if _, recordsErr := frame.Records[struct{ Missing int }](); !errors.Is(recordsErr, ErrColumnNotFound) {
		t.Fatalf("missing column error = %v", recordsErr)
	}
	if _, recordsErr := frame.Records[struct{ Value string }](); !errors.Is(recordsErr, ErrColumnType) {
		t.Fatalf("column type error = %v", recordsErr)
	}
}

func TestEmptyRecordRetainsRows(t *testing.T) {
	frame, err := FromRecords([]struct{}{{}, {}, {}})
	if err != nil {
		t.Fatal(err)
	}
	if frame.Len() != 3 || frame.Width() != 0 {
		t.Fatalf("shape = %dx%d", frame.Len(), frame.Width())
	}
	records, err := frame.Records[struct{}]()
	if err != nil {
		t.Fatal(err)
	}
	if len(records) != 3 {
		t.Fatalf("record count = %d", len(records))
	}
}

func TestRecordBackedFrameOperations_PreserveValuesAndValidity(t *testing.T) {
	type name string
	type record struct {
		ID   int
		Name *name
	}
	a := name("a")
	b := name("b")
	frame, err := FromRecords([]record{{ID: 1, Name: &a}, {ID: 2}, {ID: 3, Name: &b}})
	if err != nil {
		t.Fatal(err)
	}

	taken := frame.Take([]int{2, 0})
	sliced := frame.Slice(1, 3)
	filtered := frame.Filter(mask.New([]bool{true, false, true}))
	concatenated, err := taken.Concat(filtered)
	if err != nil {
		t.Fatal(err)
	}
	tests := []struct {
		name     string
		frame    Frame
		ids      []int
		presence []bool
	}{
		{name: "Take preserves values and validity", frame: taken, ids: []int{3, 1}, presence: []bool{true, true}},
		{name: "Slice preserves values and validity", frame: sliced, ids: []int{2, 3}, presence: []bool{false, true}},
		{name: "Filter preserves values and validity", frame: filtered, ids: []int{1, 3}, presence: []bool{true, true}},
		{name: "Concat preserves values and validity", frame: concatenated, ids: []int{3, 1, 1, 3}, presence: []bool{true, true, true, true}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			ids, columnErr := test.frame.Column[int]("ID")
			if columnErr != nil {
				t.Fatal(columnErr)
			}
			if got := ids.Values(); !slices.Equal(got, test.ids) {
				t.Fatalf("IDs = %v, want %v", got, test.ids)
			}
			records, recordsErr := test.frame.Records[record]()
			if recordsErr != nil {
				t.Fatal(recordsErr)
			}
			if len(records) != len(test.ids) {
				t.Fatalf("record count = %d", len(records))
			}
			for i, record := range records {
				if got := record.Name != nil; got != test.presence[i] {
					t.Fatalf("record %d name presence = %t, want %t", i, got, test.presence[i])
				}
			}
		})
	}
}

func TestConcatTypedAndReflectedColumns(t *testing.T) {
	type identifier int
	type record struct {
		Value *identifier
	}
	one, two := identifier(1), identifier(2)
	reflected, err := FromRecords([]record{{Value: &one}, {}})
	if err != nil {
		t.Fatal(err)
	}
	reflectedOther, err := FromRecords([]record{{}, {Value: &two}})
	if err != nil {
		t.Fatal(err)
	}
	first, second, none := series.Some(one), series.Some(two), series.None[identifier]()
	typed, err := New(ColumnFromSeries("Value", series.FromOptionals([]series.Optional[identifier]{none, second})))
	if err != nil {
		t.Fatal(err)
	}
	forward := []series.Optional[identifier]{first, none, none, second}
	tests := []struct {
		left, right Frame
		want        []series.Optional[identifier]
	}{
		{left: reflected, right: typed, want: forward},
		{left: typed, right: reflected, want: []series.Optional[identifier]{none, second, first, none}},
		{left: reflected, right: reflectedOther, want: forward},
	}
	for i, test := range tests {
		joined, concatErr := test.left.Concat(test.right)
		if concatErr != nil {
			t.Fatal(concatErr)
		}
		values, columnErr := joined.Column[identifier]("Value")
		if columnErr != nil {
			t.Fatal(columnErr)
		}
		if got := values.Optionals(); !slices.Equal(got, test.want) {
			t.Fatalf("Concat %d values = %v, want %v", i, got, test.want)
		}
	}
}

func FuzzRecordRoundTrip(f *testing.F) {
	type recordID int16
	type Metadata struct {
		Active bool
	}
	type record struct {
		Metadata
		ID       recordID
		Dynamic  any
		Name     *string
		Score    series.Optional[int]
		Optional series.Optional[any]
	}
	f.Add([]byte{0, 1, 2, 3, 7, 15, 31, 63, 127, 255})
	f.Add([]byte{})
	f.Fuzz(func(t *testing.T, data []byte) {
		data = data[:min(len(data), 64)]
		records := make([]record, len(data))
		names := [...]string{"", "alpha", "beta"}
		for i, value := range data {
			records[i].Active = value&1 != 0
			records[i].ID = recordID(int8(value))
			switch (value >> 1) % 3 {
			case 1:
				records[i].Dynamic = int(int8(value))
			case 2:
				records[i].Dynamic = names[int(value)%len(names)]
			}
			if value&4 != 0 {
				name := names[int(value>>2)%len(names)]
				records[i].Name = &name
			}
			if value&8 != 0 {
				records[i].Score = series.Some(int(int8(value)))
			}
			switch (value >> 4) % 3 {
			case 1:
				records[i].Optional = series.Some[any](nil)
			case 2:
				records[i].Optional = series.Some[any](names[int(value)%len(names)])
			}
		}

		frame, err := FromRecords(records)
		if err != nil {
			t.Fatal(err)
		}
		got, err := frame.Records[record]()
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(got, records) {
			t.Fatalf("round trip = %#v, want %#v", got, records)
		}
	})
}

func assertTypedRecordColumn[T any](t *testing.T, frame Frame, name string) {
	t.Helper()
	index := slices.IndexFunc(frame.columns, func(column column) bool {
		return column.name == name
	})
	if index < 0 {
		t.Fatalf("column %q not found", name)
	}
	if _, ok := frame.columns[index].values.(typedData[T]); !ok {
		t.Fatalf("column %q storage is %T, want typedData[%v]", name, frame.columns[index].values, reflect.TypeFor[T]())
	}
}

func BenchmarkFrameRows(b *testing.B) {
	frame, err := New(Column("value", make([]int, 10_000)))
	if err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	var result Row
	for b.Loop() {
		for _, row := range frame.Rows() {
			result = row
		}
	}
	runtime.KeepAlive(result)
}

func BenchmarkFrameRow(b *testing.B) {
	frame, err := New(Column("value", []int{1}))
	if err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	var result Row
	for b.Loop() {
		result = frame.Row(0)
	}
	runtime.KeepAlive(result)
}

func BenchmarkFromRecords(b *testing.B) {
	type record struct {
		ID    int
		Name  string
		Score *float64
	}
	value := 1.5
	records := make([]record, 10_000)
	for i := range records {
		records[i] = record{ID: i, Name: "value", Score: &value}
	}
	b.ReportAllocs()
	var result Frame
	for b.Loop() {
		var recordsErr error
		result, recordsErr = FromRecords(records)
		if recordsErr != nil {
			b.Fatal(recordsErr)
		}
	}
	runtime.KeepAlive(result)
}

func BenchmarkRecords(b *testing.B) {
	type record struct {
		ID   int
		Name string
	}
	frame, err := New(Column("ID", make([]int, 10_000)), Column("Name", make([]string, 10_000)))
	if err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	var result []record
	for b.Loop() {
		var recordsErr error
		result, recordsErr = frame.Records[record]()
		if recordsErr != nil {
			b.Fatal(recordsErr)
		}
	}
	runtime.KeepAlive(result)
}

func BenchmarkRecordBackedFrameColumn(b *testing.B) {
	type record struct {
		Value    int
		Nullable *int
	}

	const size = 10_000
	one := 1
	records := make([]record, size)
	for i := 0; i < len(records); i += 2 {
		records[i].Nullable = &one
	}
	frame, err := FromRecords(records)
	if err != nil {
		b.Fatal(err)
	}
	reflected := Frame{columns: slices.Clone(frame.columns), rowCount: frame.Len()}
	for i, stored := range frame.columns {
		values := stored.values.(typedData[int]).values
		data := reflectData{values: reflect.ValueOf(values.Values())}
		if values.Nullable() {
			data.validity = bitmap.FromBools(values.Validity())
		}
		reflected.columns[i].values = data
	}

	for _, benchmark := range []struct {
		name   string
		frame  Frame
		column string
	}{
		{name: "NonNull/Optimized", frame: frame, column: "Value"},
		{name: "NonNull/Reference", frame: reflected, column: "Value"},
		{name: "Nullable/Optimized", frame: frame, column: "Nullable"},
		{name: "Nullable/Reference", frame: reflected, column: "Nullable"},
	} {
		b.Run(benchmark.name, func(b *testing.B) {
			b.ReportAllocs()
			var result series.Series[int]
			for b.Loop() {
				var columnErr error
				result, columnErr = benchmark.frame.Column[int](benchmark.column)
				if columnErr != nil {
					b.Fatal(columnErr)
				}
			}
			runtime.KeepAlive(result)
		})
	}
}

func BenchmarkRecordBackedFrameFilter(b *testing.B) {
	type record struct {
		ID   int
		Name string
	}
	frame, err := FromRecords(make([]record, 10_000))
	if err != nil {
		b.Fatal(err)
	}
	reflected := Frame{columns: slices.Clone(frame.columns), rowCount: frame.Len()}
	for i, stored := range frame.columns {
		values := reflect.MakeSlice(reflect.SliceOf(stored.typeOf), frame.Len(), frame.Len())
		for row := range frame.Len() {
			value, _ := stored.values.at(row)
			values.Index(row).Set(reflect.ValueOf(value))
		}
		reflected.columns[i].values = reflectData{values: values}
	}
	selection := mask.NewFunc(frame.Len(), func(i int) bool { return i%2 == 0 })
	for _, benchmark := range []struct {
		name  string
		frame Frame
	}{
		{name: "Optimized", frame: frame},
		{name: "Reference", frame: reflected},
	} {
		b.Run(benchmark.name, func(b *testing.B) {
			b.ReportAllocs()
			var result Frame
			for b.Loop() {
				result = benchmark.frame.Filter(selection)
			}
			runtime.KeepAlive(result)
		})
	}
}

func BenchmarkRecordBackedRowGet(b *testing.B) {
	type record struct {
		Value int
	}
	frame, err := FromRecords(make([]record, 10_000))
	if err != nil {
		b.Fatal(err)
	}
	typed := frame.columns[0].values.(typedData[int]).values
	reflected := Frame{
		columns:  []column{{name: "Value", typeOf: reflect.TypeFor[int](), values: reflectData{values: reflect.ValueOf(typed.Values())}}},
		rowCount: frame.Len(),
	}
	for _, benchmark := range []struct {
		name string
		row  Row
	}{
		{name: "Optimized", row: frame.Row(frame.Len() / 2)},
		{name: "Reference", row: reflected.Row(reflected.Len() / 2)},
	} {
		b.Run(benchmark.name, func(b *testing.B) {
			b.ReportAllocs()
			var value int
			var present bool
			for b.Loop() {
				var getErr error
				value, present, getErr = benchmark.row.Get[int]("Value")
				if getErr != nil {
					b.Fatal(getErr)
				}
			}
			runtime.KeepAlive(value)
			runtime.KeepAlive(present)
		})
	}
}
