package dataframe

import (
	"errors"
	"hash/maphash"
	"reflect"
	"slices"
	"testing"

	"github.com/joeychilson/dataframe/series"
)

func TestGroupByOrderKeysSizesAndGroups(t *testing.T) {
	keys := series.FromOptionals([]series.Optional[string]{
		series.Some("b"),
		series.None[string](),
		series.Some("a"),
		series.Some("b"),
		series.None[string](),
	})
	frame, err := New(Column("id", []int{0, 1, 2, 3, 4}))
	if err != nil {
		t.Fatal(err)
	}
	grouped := frame.GroupBy(keys)
	if grouped.Len() != 3 {
		t.Fatalf("Len() = %d, want 3", grouped.Len())
	}
	if got := grouped.Keys().Optionals(); !reflect.DeepEqual(got, []series.Optional[string]{series.Some("b"), series.None[string](), series.Some("a")}) {
		t.Fatalf("Keys() = %#v", got)
	}
	if got := grouped.Sizes().Values(); !slices.Equal(got, []int{2, 2, 1}) {
		t.Fatalf("Sizes() = %v", got)
	}

	var groupRows [][]int
	for _, group := range grouped.Groups() {
		ids, err := group.Column[int]("id")
		if err != nil {
			t.Fatal(err)
		}
		groupRows = append(groupRows, ids.Values())
	}
	wantRows := [][]int{{0, 3}, {1, 4}, {2}}
	if !reflect.DeepEqual(groupRows, wantRows) {
		t.Fatalf("Groups rows = %v, want %v", groupRows, wantRows)
	}
}

func TestGroupedAggregations(t *testing.T) {
	frame, err := New(Column("id", []int{0, 1, 2, 3, 4}))
	if err != nil {
		t.Fatal(err)
	}
	grouped := frame.GroupBy(series.New([]string{"a", "b", "a", "b", "c"}))
	values := series.FromOptionals([]series.Optional[int]{
		series.Some(2),
		series.None[int](),
		series.Some(4),
		series.Some(8),
		series.None[int](),
	})

	if got := grouped.Sum(values).Optionals(); !reflect.DeepEqual(got, []series.Optional[int]{series.Some(6), series.Some(8), series.None[int]()}) {
		t.Fatalf("Sum = %#v", got)
	}
	if got := grouped.Mean(values).Optionals(); !reflect.DeepEqual(got, []series.Optional[float64]{series.Some(3.0), series.Some(8.0), series.None[float64]()}) {
		t.Fatalf("Mean = %#v", got)
	}
	if got := grouped.Min(values).Optionals(); !reflect.DeepEqual(got, []series.Optional[int]{series.Some(2), series.Some(8), series.None[int]()}) {
		t.Fatalf("Min = %#v", got)
	}
	if got := grouped.Max(values).Optionals(); !reflect.DeepEqual(got, []series.Optional[int]{series.Some(4), series.Some(8), series.None[int]()}) {
		t.Fatalf("Max = %#v", got)
	}
	if got := grouped.Count(values).Values(); !slices.Equal(got, []int{2, 1, 0}) {
		t.Fatalf("Count = %v", got)
	}
	if got := grouped.FirstPresent(values).Optionals(); !reflect.DeepEqual(got, []series.Optional[int]{series.Some(2), series.Some(8), series.None[int]()}) {
		t.Fatalf("FirstPresent = %#v", got)
	}
	if got := grouped.LastPresent(values).Optionals(); !reflect.DeepEqual(got, []series.Optional[int]{series.Some(4), series.Some(8), series.None[int]()}) {
		t.Fatalf("LastPresent = %#v", got)
	}
}

func TestGroupedMapTryMapAndResult(t *testing.T) {
	frame, err := New(Column("value", []int{1, 2, 3}))
	if err != nil {
		t.Fatal(err)
	}
	grouped := frame.GroupBy(series.New([]string{"a", "a", "b"}))
	values, _ := frame.Column[int]("value")
	mapped := grouped.Map(values, func(group series.Series[int]) (int, bool) {
		return group.Len(), true
	})
	if !slices.Equal(mapped.Values(), []int{2, 1}) || mapped.NullCount() != 0 {
		t.Fatalf("Map result = %#v", mapped.Optionals())
	}

	wantErr := errors.New("stop")
	calls := 0
	_, err = grouped.TryMap(values, func(series.Series[int]) (int, bool, error) {
		calls++
		if calls == 2 {
			return 0, false, wantErr
		}
		return 1, true, nil
	})
	if !errors.Is(err, wantErr) || calls != 2 {
		t.Fatalf("TryMap error = %v, calls = %d", err, calls)
	}

	result, err := grouped.Result("group", ColumnFromSeries("total", grouped.Sum(values)))
	if err != nil {
		t.Fatal(err)
	}
	if got := result.Names(); !slices.Equal(got, []string{"group", "total"}) {
		t.Fatalf("Result names = %v", got)
	}
	if _, err := grouped.Result("group", Column("bad", []int{1})); !errors.Is(err, ErrRowCount) {
		t.Fatalf("Result row error = %v", err)
	}
}

func TestGroupByUsing(t *testing.T) {
	frame, err := New(Column("id", []int{0, 1, 2}))
	if err != nil {
		t.Fatal(err)
	}
	keys := series.New([][]int{{1, 2}, {1, 2}, {3}})
	grouped := frame.GroupByUsing(keys, intSliceHasher{})
	if grouped.Len() != 2 || !slices.Equal(grouped.Sizes().Values(), []int{2, 1}) {
		t.Fatalf("groups = %d, sizes = %v", grouped.Len(), grouped.Sizes().Values())
	}
	assertPanics(t, func() { frame.GroupBy(series.New([]int{1})) })
}

func TestGroupedOperationsPanicOnLengthMismatch(t *testing.T) {
	frame, err := New(Column("value", []int{1, 2}))
	if err != nil {
		t.Fatal(err)
	}
	grouped := frame.GroupBy(series.New([]int{1, 1}))
	short := series.New([]int{1})
	tests := []struct {
		name string
		call func()
	}{
		{name: "Sum", call: func() { grouped.Sum(short) }},
		{name: "Mean", call: func() { grouped.Mean(short) }},
		{name: "Min", call: func() { grouped.Min(short) }},
		{name: "Max", call: func() { grouped.Max(short) }},
		{name: "Count", call: func() { grouped.Count(short) }},
		{name: "FirstPresent", call: func() { grouped.FirstPresent(short) }},
		{name: "LastPresent", call: func() { grouped.LastPresent(short) }},
		{name: "Map", call: func() { grouped.Map(short, series.Sum[int]) }},
		{name: "TryMap", call: func() {
			_, _ = grouped.TryMap(short, func(series.Series[int]) (int, bool, error) { return 0, false, nil })
		}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			assertPanics(t, test.call)
		})
	}
}

func BenchmarkGroupBySum(b *testing.B) {
	keys := make([]int, 10_000)
	values := make([]float64, len(keys))
	for i := range keys {
		keys[i] = i % 100
		values[i] = float64(i)
	}
	frame, err := New(Column("value", values))
	if err != nil {
		b.Fatal(err)
	}
	key := series.New(keys)
	value := series.New(values)
	b.ReportAllocs()
	for b.Loop() {
		_ = frame.GroupBy(key).Sum(value)
	}
}

type intSliceHasher struct{}

func (intSliceHasher) Hash(hash *maphash.Hash, values []int) {
	maphash.WriteComparable(hash, len(values))
	for _, value := range values {
		maphash.WriteComparable(hash, value)
	}
}

func (intSliceHasher) Equal(left, right []int) bool {
	return slices.Equal(left, right)
}
