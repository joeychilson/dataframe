package dataframe

import (
	"errors"
	"hash/maphash"
	"reflect"
	"slices"
	"testing"

	"github.com/joeychilson/dataframe/mask"
	"github.com/joeychilson/dataframe/series"
)

func TestNewAndSchema(t *testing.T) {
	notes, err := series.NewNullable([]string{"one", ""}, []bool{true, false})
	if err != nil {
		t.Fatal(err)
	}
	frame, err := New(Column("id", []int{1, 2}), ColumnFromSeries("note", notes))
	if err != nil {
		t.Fatal(err)
	}

	if frame.Len() != 2 || frame.Width() != 2 {
		t.Fatalf("shape = %dx%d, want 2x2", frame.Len(), frame.Width())
	}
	if got := frame.Names(); !slices.Equal(got, []string{"id", "note"}) {
		t.Fatalf("Names() = %v", got)
	}
	wantSchema := []Field{
		{Name: "id", Type: reflect.TypeFor[int]()},
		{Name: "note", Type: reflect.TypeFor[string](), Nullable: true},
	}
	if got := frame.Schema(); !reflect.DeepEqual(got, wantSchema) {
		t.Fatalf("Schema() = %#v, want %#v", got, wantSchema)
	}
	if !frame.Has("id") || frame.Has("missing") {
		t.Fatalf("Has returned an unexpected result")
	}
	if got := frame.String(); got != "Frame[2x2]{id:int, note:string?}" {
		t.Fatalf("String() = %q", got)
	}

	// Returned schema slices must not mutate the frame.
	names := frame.Names()
	names[0] = "changed"
	fields := frame.Schema()
	fields[0].Name = "changed"
	if !frame.Has("id") {
		t.Fatal("Names or Schema exposed frame storage")
	}
}

func TestNewErrors(t *testing.T) {
	tests := []struct {
		name    string
		columns []ColumnSpec
		want    error
	}{
		{name: "nil", columns: []ColumnSpec{nil}, want: ErrSchemaMismatch},
		{name: "empty name", columns: []ColumnSpec{Column("", []int{1})}, want: ErrInvalidName},
		{name: "duplicate", columns: []ColumnSpec{Column("x", []int{1}), Column("x", []int{2})}, want: ErrColumnConflict},
		{name: "row count", columns: []ColumnSpec{Column("x", []int{1}), Column("y", []int{1, 2})}, want: ErrRowCount},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, err := New(test.columns...)
			if !errors.Is(err, test.want) {
				t.Fatalf("New error = %v, want %v", err, test.want)
			}
		})
	}
}

func TestFrameColumnAndWith(t *testing.T) {
	frame, err := New(Column("id", []int{1, 2}))
	if err != nil {
		t.Fatal(err)
	}
	ids, err := frame.Column[int]("id")
	if err != nil {
		t.Fatal(err)
	}
	if got := ids.Values(); !slices.Equal(got, []int{1, 2}) {
		t.Fatalf("Column = %v", got)
	}
	if _, err := frame.Column[string]("id"); !errors.Is(err, ErrColumnType) {
		t.Fatalf("type error = %v", err)
	}
	if _, err := frame.Column[int]("missing"); !errors.Is(err, ErrColumnNotFound) {
		t.Fatalf("missing error = %v", err)
	}

	frame, err = frame.WithValues("name", []string{"a", "b"})
	if err != nil {
		t.Fatal(err)
	}
	frame, err = frame.WithValues("id", []int64{10, 20})
	if err != nil {
		t.Fatal(err)
	}
	if got := frame.Names(); !slices.Equal(got, []string{"id", "name"}) {
		t.Fatalf("names after replacement = %v", got)
	}
	if _, err := frame.Column[int64]("id"); err != nil {
		t.Fatalf("replacement column: %v", err)
	}
	if _, err := frame.WithValues("bad", []int{1}); !errors.Is(err, ErrRowCount) {
		t.Fatalf("row-count error = %v", err)
	}
	if _, err := frame.WithValues("", []int{1, 2}); !errors.Is(err, ErrInvalidName) {
		t.Fatalf("name error = %v", err)
	}
}

func TestFrameSchemaTransforms(t *testing.T) {
	frame, err := New(
		Column("a", []int{1, 2}),
		Column("b", []string{"x", "y"}),
		Column("c", []bool{true, false}),
	)
	if err != nil {
		t.Fatal(err)
	}

	selected, err := frame.Select("c", "a")
	if err != nil {
		t.Fatal(err)
	}
	if got := selected.Names(); !slices.Equal(got, []string{"c", "a"}) {
		t.Fatalf("Select names = %v", got)
	}
	empty, err := frame.Select()
	if err != nil {
		t.Fatal(err)
	}
	if empty.Width() != 0 || empty.Len() != 2 {
		t.Fatalf("empty selection shape = %dx%d", empty.Len(), empty.Width())
	}

	renamed, err := frame.Rename("b", "name")
	if err != nil {
		t.Fatal(err)
	}
	if got := renamed.Names(); !slices.Equal(got, []string{"a", "name", "c"}) {
		t.Fatalf("Rename names = %v", got)
	}
	if _, err := frame.Rename("a", "b"); !errors.Is(err, ErrColumnConflict) {
		t.Fatalf("rename conflict = %v", err)
	}

	dropped, err := frame.Drop("a", "c", "a")
	if err != nil {
		t.Fatal(err)
	}
	if got := dropped.Names(); !slices.Equal(got, []string{"b"}) {
		t.Fatalf("Drop names = %v", got)
	}
	if _, err := frame.Drop("missing"); !errors.Is(err, ErrColumnNotFound) {
		t.Fatalf("drop error = %v", err)
	}
	if _, err := frame.Select("a", "a"); !errors.Is(err, ErrColumnConflict) {
		t.Fatalf("duplicate select error = %v", err)
	}
}

func TestFrameRowsAndFiltering(t *testing.T) {
	frame, err := New(Column("n", []int{10, 20, 30, 40}))
	if err != nil {
		t.Fatal(err)
	}

	assertInts := func(t *testing.T, frame Frame, want []int) {
		t.Helper()
		values, err := frame.Column[int]("n")
		if err != nil {
			t.Fatal(err)
		}
		if got := values.Values(); !slices.Equal(got, want) {
			t.Fatalf("values = %v, want %v", got, want)
		}
	}
	assertInts(t, frame.Take([]int{3, 1, 1}), []int{40, 20, 20})
	assertInts(t, frame.Head(2), []int{10, 20})
	assertInts(t, frame.Tail(2), []int{30, 40})
	assertInts(t, frame.Slice(1, 3), []int{20, 30})
	assertInts(t, frame.Filter(mask.New([]bool{true, false, true, false})), []int{10, 30})
	assertInts(t, frame.FilterFunc(series.New([]int{1, 2, 3, 4}), func(value int) bool {
		return value%2 == 0
	}), []int{20, 40})

	zeroWidth, err := frame.Select()
	if err != nil {
		t.Fatal(err)
	}
	if got := zeroWidth.Take([]int{3, 0, 0}); got.Len() != 3 || got.Width() != 0 {
		t.Fatalf("zero-width Take shape = %dx%d", got.Len(), got.Width())
	}
	assertPanics(t, func() { zeroWidth.Take([]int{4}) })
	assertPanics(t, func() { frame.Filter(mask.All(3)) })
	assertPanics(t, func() { frame.Head(-1) })
}

func TestFrameDistinct(t *testing.T) {
	names, err := series.NewNullable([]string{"a", "a", "", "", "b"}, []bool{true, true, false, false, true})
	if err != nil {
		t.Fatal(err)
	}
	frame, err := New(Column("n", []int{1, 1, 1, 1, 2}), ColumnFromSeries("name", names))
	if err != nil {
		t.Fatal(err)
	}
	distinct, err := frame.Distinct()
	if err != nil {
		t.Fatal(err)
	}
	if distinct.Len() != 3 {
		t.Fatalf("Distinct length = %d, want 3", distinct.Len())
	}

	by := frame.DistinctBy(series.New([]int{1, 1, 2, 2, 1}))
	if got, _ := by.Column[int]("n"); !slices.Equal(got.Values(), []int{1, 1}) {
		t.Fatalf("DistinctBy values = %v", got.Values())
	}
	using := frame.DistinctByUsing(series.New([]int{1, 1, 2, 2, 1}), maphash.ComparableHasher[int]{})
	if using.Len() != 2 {
		t.Fatalf("DistinctByUsing length = %d", using.Len())
	}

	unsupported, err := New(Column("slice", [][]int{{1}, {1}}))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := unsupported.Distinct(); !errors.Is(err, ErrUnsupported) {
		t.Fatalf("Distinct error = %v", err)
	}
	dynamic, err := New(Column[any]("dynamic", []any{[]int{1}, []int{1}}))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := dynamic.Distinct(); !errors.Is(err, ErrUnsupported) {
		t.Fatalf("dynamic Distinct error = %v", err)
	}

	type nestedDynamic struct {
		Value any
	}
	nested, err := New(Column("nested", []nestedDynamic{{Value: []int{1}}, {Value: []int{1}}}))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := nested.Distinct(); !errors.Is(err, ErrUnsupported) {
		t.Fatalf("nested dynamic Distinct error = %v", err)
	}
}

func TestFrameConcat(t *testing.T) {
	left, err := New(Column("id", []int{1}), Column("name", []string{"a"}))
	if err != nil {
		t.Fatal(err)
	}
	nullable, err := series.NewNullable([]string{"", "c"}, []bool{false, true})
	if err != nil {
		t.Fatal(err)
	}
	right, err := New(Column("id", []int{2, 3}), ColumnFromSeries("name", nullable))
	if err != nil {
		t.Fatal(err)
	}
	joined, err := left.Concat(right)
	if err != nil {
		t.Fatal(err)
	}
	if joined.Len() != 3 || !joined.Schema()[1].Nullable {
		t.Fatalf("Concat shape/schema = %dx%d %#v", joined.Len(), joined.Width(), joined.Schema())
	}
	names, err := joined.Column[string]("name")
	if err != nil {
		t.Fatal(err)
	}
	if got := names.Validity(); !slices.Equal(got, []bool{true, false, true}) {
		t.Fatalf("Concat validity = %v", got)
	}

	bad, err := New(Column("other", []int{1}))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := left.Concat(bad); !errors.Is(err, ErrSchemaMismatch) {
		t.Fatalf("Concat error = %v", err)
	}
}

func BenchmarkFrameFilter(b *testing.B) {
	values := make([]int, 10_000)
	frame, err := New(Column("value", values))
	if err != nil {
		b.Fatal(err)
	}
	selection := mask.NewFunc(len(values), func(i int) bool { return i%2 == 0 })
	b.ReportAllocs()
	for b.Loop() {
		_ = frame.Filter(selection)
	}
}

func BenchmarkFrameDistinctBy(b *testing.B) {
	values := make([]int, 10_000)
	for i := range values {
		values[i] = i % 100
	}
	frame, err := New(Column("value", values))
	if err != nil {
		b.Fatal(err)
	}
	key := series.New(values)
	b.ReportAllocs()
	for b.Loop() {
		_ = frame.DistinctBy(key)
	}
}

func assertPanics(t *testing.T, fn func()) {
	t.Helper()
	defer func() {
		if recover() == nil {
			t.Fatal("function did not panic")
		}
	}()
	fn()
}
