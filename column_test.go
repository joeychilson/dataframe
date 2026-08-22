package dataframe

import (
	"reflect"
	"slices"
	"testing"

	"github.com/joeychilson/dataframe/series"
)

func TestColumns(t *testing.T) {
	names, err := series.NewNullable([]string{"a", ""}, []bool{true, false})
	if err != nil {
		t.Fatal(err)
	}
	frame, err := New(Column("id", []int{1, 2}), ColumnFromSeries("name", names))
	if err != nil {
		t.Fatal(err)
	}

	views := slices.Collect(frame.Columns())
	if len(views) != 2 {
		t.Fatalf("Columns length = %d", len(views))
	}
	if views[0].Name() != "id" || views[0].Type() != reflect.TypeFor[int]() || views[0].Nullable() || views[0].Len() != 2 {
		t.Fatalf("first view = name %q, type %v, nullable %v, len %d", views[0].Name(), views[0].Type(), views[0].Nullable(), views[0].Len())
	}
	if value, present := views[0].At(1); value != 2 || !present {
		t.Fatalf("first view At(1) = %v, %v", value, present)
	}
	if value, present := views[1].At(1); value != nil || present {
		t.Fatalf("nullable view At(1) = %v, %v", value, present)
	}
	assertPanics(t, func() { views[0].At(2) })

	count := 0
	for range frame.Columns() {
		count++
		break
	}
	if count != 1 {
		t.Fatalf("early-stop count = %d", count)
	}
}

func TestZeroColumnView(t *testing.T) {
	var view ColumnView
	if view.Name() != "" || view.Len() != 0 || view.Type() != nil || view.Nullable() {
		t.Fatalf("zero ColumnView = {Name:%q Len:%d Type:%v Nullable:%t}", view.Name(), view.Len(), view.Type(), view.Nullable())
	}
	assertPanics(t, func() { view.At(0) })
}

func TestColumnCopiesInput(t *testing.T) {
	values := []int{1, 2}
	frame, err := New(Column("id", values))
	if err != nil {
		t.Fatal(err)
	}
	values[0] = 99
	column, err := frame.Column[int]("id")
	if err != nil {
		t.Fatal(err)
	}
	if got := column.Values(); !slices.Equal(got, []int{1, 2}) {
		t.Fatalf("column values = %v", got)
	}
}

func BenchmarkColumns(b *testing.B) {
	frame, err := New(Column("id", []int{1, 2, 3}), Column("name", []string{"a", "b", "c"}))
	if err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	for b.Loop() {
		for view := range frame.Columns() {
			_, _ = view.At(0)
		}
	}
}
