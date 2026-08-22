package dataframe

import (
	"errors"
	"reflect"
	"slices"
	"testing"

	"github.com/joeychilson/dataframe/mask"
	"github.com/joeychilson/dataframe/series"
)

func TestRowsAndRowGet(t *testing.T) {
	names := series.FromOptionals([]series.Optional[string]{series.Some("a"), series.None[string]()})
	frame, err := New(Column("id", []int{1, 2}), ColumnFromSeries("name", names))
	if err != nil {
		t.Fatal(err)
	}

	var ids []int
	for index, row := range frame.Rows() {
		id, present, err := row.Get[int]("id")
		if err != nil || !present {
			t.Fatalf("row %d id = %d, %v, %v", index, id, present, err)
		}
		ids = append(ids, id)
	}
	if !slices.Equal(ids, []int{1, 2}) {
		t.Fatalf("row ids = %v", ids)
	}
	if value, present, err := frame.Row(1).Get[string]("name"); err != nil || present || value != "" {
		t.Fatalf("null name = %q, %v, %v", value, present, err)
	}
	if _, _, err := frame.Row(0).Get[string]("id"); !errors.Is(err, ErrColumnType) {
		t.Fatalf("type error = %v", err)
	}
	if _, _, err := frame.Row(0).Get[int]("missing"); !errors.Is(err, ErrColumnNotFound) {
		t.Fatalf("missing error = %v", err)
	}
	assertPanics(t, func() { frame.Row(2) })

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

func TestFromRecordsAndRecords(t *testing.T) {
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
		{Metadata: Metadata{Active: true}, ID: 1, Name: &a, Score: series.Some(1.5), Ignored: "x"},
		{ID: 2, Score: series.None[float64](), Ignored: "y"},
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

func TestRecordErrors(t *testing.T) {
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
	if _, err := frame.Records[struct{ Value int }](); !errors.Is(err, ErrInvalidRecord) {
		t.Fatalf("null scalar error = %v", err)
	}
	if _, err := frame.Records[struct{ Missing int }](); !errors.Is(err, ErrColumnNotFound) {
		t.Fatalf("missing column error = %v", err)
	}
	if _, err := frame.Records[struct{ Value string }](); !errors.Is(err, ErrColumnType) {
		t.Fatalf("column type error = %v", err)
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

func TestRecordBackedFrameOperations(t *testing.T) {
	type record struct {
		ID   int
		Name *string
	}
	a := "a"
	b := "b"
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
		name  string
		frame Frame
		ids   []int
	}{
		{name: "Take", frame: taken, ids: []int{3, 1}},
		{name: "Slice", frame: sliced, ids: []int{2, 3}},
		{name: "Filter", frame: filtered, ids: []int{1, 3}},
		{name: "Concat", frame: concatenated, ids: []int{3, 1, 1, 3}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			ids, err := test.frame.Column[int]("ID")
			if err != nil {
				t.Fatal(err)
			}
			if got := ids.Values(); !slices.Equal(got, test.ids) {
				t.Fatalf("IDs = %v, want %v", got, test.ids)
			}
			records, err := test.frame.Records[record]()
			if err != nil {
				t.Fatal(err)
			}
			if len(records) != len(test.ids) {
				t.Fatalf("record count = %d", len(records))
			}
		})
	}
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
	for b.Loop() {
		if _, err := FromRecords(records); err != nil {
			b.Fatal(err)
		}
	}
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
	for b.Loop() {
		if _, err := frame.Records[record](); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkRowGetRecordColumn(b *testing.B) {
	type record struct {
		Value int
	}
	frame, err := FromRecords(make([]record, 10_000))
	if err != nil {
		b.Fatal(err)
	}
	row := frame.Row(frame.Len() / 2)
	b.ReportAllocs()
	for b.Loop() {
		if _, _, err := row.Get[int]("Value"); err != nil {
			b.Fatal(err)
		}
	}
}
