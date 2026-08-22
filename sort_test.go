package dataframe

import (
	"cmp"
	"slices"
	"testing"

	"github.com/joeychilson/dataframe/series"
)

func TestSortedBy(t *testing.T) {
	frame, err := New(
		Column("id", []int{0, 1, 2, 3, 4}),
		Column("group", []string{"b", "a", "a", "b", "a"}),
		Column("value", []int{2, 2, 1, 1, 2}),
	)
	if err != nil {
		t.Fatal(err)
	}
	groups, _ := frame.Column[string]("group")
	values, _ := frame.Column[int]("value")
	sorted := frame.SortedBy(Asc(groups), Desc(values))
	ids, err := sorted.Column[int]("id")
	if err != nil {
		t.Fatal(err)
	}
	if got := ids.Values(); !slices.Equal(got, []int{1, 4, 2, 0, 3}) {
		t.Fatalf("sorted ids = %v", got)
	}

	// Equal keys retain their input order.
	stable := frame.SortedBy(ByFunc(values, func(_, _ int) int { return 0 }))
	stableIDs, _ := stable.Column[int]("id")
	if got := stableIDs.Values(); !slices.Equal(got, []int{0, 1, 2, 3, 4}) {
		t.Fatalf("stable ids = %v", got)
	}
}

func TestSortNullPlacementAndReverse(t *testing.T) {
	key := series.FromOptionals([]series.Optional[int]{series.Some(2), series.None[int](), series.Some(1)})
	frame, err := New(Column("id", []int{0, 1, 2}))
	if err != nil {
		t.Fatal(err)
	}
	tests := []struct {
		name string
		key  SortKey
		want []int
	}{
		{name: "ascending nulls last", key: Asc(key), want: []int{2, 0, 1}},
		{name: "descending nulls last", key: Desc(key), want: []int{0, 2, 1}},
		{name: "ascending nulls first", key: Asc(key).NullsFirst(), want: []int{1, 2, 0}},
		{name: "reverse retains null placement", key: Asc(key).NullsFirst().Reverse(), want: []int{1, 0, 2}},
		{name: "last call wins", key: Asc(key).NullsFirst().NullsLast(), want: []int{2, 0, 1}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			sorted := frame.SortedBy(test.key)
			ids, _ := sorted.Column[int]("id")
			if got := ids.Values(); !slices.Equal(got, test.want) {
				t.Fatalf("ids = %v, want %v", got, test.want)
			}
		})
	}
}

func TestSortedByPanicsOnInvalidKey(t *testing.T) {
	frame, err := New(Column("id", []int{1, 2}))
	if err != nil {
		t.Fatal(err)
	}
	assertPanics(t, func() { frame.SortedBy(Asc(series.New([]int{1}))) })
	assertPanics(t, func() { frame.SortedBy(SortKey{}) })
	assertPanics(t, func() { ByFunc(series.New([]int{1}), nil) })
}

func BenchmarkSortedBy(b *testing.B) {
	values := make([]int, 10_000)
	for i := range values {
		values[i] = len(values) - i
	}
	frame, err := New(Column("value", values))
	if err != nil {
		b.Fatal(err)
	}
	key := ByFunc(series.New(values), cmp.Compare[int])
	b.ReportAllocs()
	for b.Loop() {
		_ = frame.SortedBy(key)
	}
}
