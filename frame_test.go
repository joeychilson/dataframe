package dataframe_test

import (
	"cmp"
	"errors"
	"reflect"
	"slices"
	"strconv"
	"testing"

	dataframe "github.com/joeychilson/dataframe"
	"github.com/joeychilson/dataframe/series"
)

type seasonKey struct {
	Team string
	Year int
}

func TestFrameGenericMethods(t *testing.T) {
	frame, err := dataframe.New().WithColumn("age", []int{17, 25, 40})
	if err != nil {
		t.Fatal(err)
	}
	frame, err = frame.WithColumn("name", []string{"Ada", "Ben", "Cy"})
	if err != nil {
		t.Fatal(err)
	}

	frame, err = frame.Derive("age", "label", func(age int) string {
		return strconv.Itoa(age) + " years"
	})
	if err != nil {
		t.Fatal(err)
	}

	labels, err := frame.Column[string]("label")
	if err != nil {
		t.Fatal(err)
	}
	if got, want := labels.Values(), []string{"17 years", "25 years", "40 years"}; !slices.Equal(got, want) {
		t.Fatalf("labels = %v, want %v", got, want)
	}

	adults, err := frame.Filter("age", func(age int) bool { return age >= 18 })
	if err != nil {
		t.Fatal(err)
	}
	names, err := adults.Column[string]("name")
	if err != nil {
		t.Fatal(err)
	}
	if got, want := names.Values(), []string{"Ben", "Cy"}; !slices.Equal(got, want) {
		t.Fatalf("names = %v, want %v", got, want)
	}
}

func TestDerive2BuildsCompositeKeysWithInferredTypes(t *testing.T) {
	frame, err := dataframe.New().WithColumn("team", []string{"red", "blue", "red"})
	if err != nil {
		t.Fatal(err)
	}
	frame, err = frame.WithColumn("year", []int{2025, 2025, 2026})
	if err != nil {
		t.Fatal(err)
	}

	derived, err := frame.Derive2("team", "year", "season", func(team string, year int) seasonKey {
		return seasonKey{Team: team, Year: year}
	})
	if err != nil {
		t.Fatal(err)
	}
	seasons, err := derived.Column[seasonKey]("season")
	if err != nil {
		t.Fatal(err)
	}
	want := []seasonKey{
		{Team: "red", Year: 2025},
		{Team: "blue", Year: 2025},
		{Team: "red", Year: 2026},
	}
	if got := seasons.Values(); !slices.Equal(got, want) {
		t.Fatalf("seasons = %v, want %v", got, want)
	}
	if seasons.Nullable() {
		t.Fatal("dense sources produced a nullable derived column")
	}
}

func TestDerive2PropagatesNullsFromEitherSource(t *testing.T) {
	teams, err := series.NewNullable(
		[]string{"red", "missing", "blue"},
		[]bool{true, false, true},
	)
	if err != nil {
		t.Fatal(err)
	}
	years, err := series.NewNullable(
		[]int{2025, 2025, 0},
		[]bool{true, true, false},
	)
	if err != nil {
		t.Fatal(err)
	}
	frame, err := dataframe.New().WithSeries("team", teams)
	if err != nil {
		t.Fatal(err)
	}
	frame, err = frame.WithSeries("year", years)
	if err != nil {
		t.Fatal(err)
	}

	calls := 0
	derived, err := frame.Derive2("team", "year", "season", func(team string, year int) seasonKey {
		calls++
		return seasonKey{Team: team, Year: year}
	})
	if err != nil {
		t.Fatal(err)
	}
	seasons, err := derived.Column[seasonKey]("season")
	if err != nil {
		t.Fatal(err)
	}
	if calls != 1 {
		t.Fatalf("callback calls = %d, want 1", calls)
	}
	if got, want := seasons.Validity(), []bool{true, false, false}; !slices.Equal(got, want) {
		t.Fatalf("validity = %v, want %v", got, want)
	}
	if got, valid := seasons.At(0); !valid || got != (seasonKey{Team: "red", Year: 2025}) {
		t.Fatalf("row 0 = (%v, %v), want ({red 2025}, true)", got, valid)
	}

	if _, err := frame.Derive2("team", "year", "invalid", func(team int, year int) seasonKey {
		return seasonKey{Year: team + year}
	}); !errors.Is(err, dataframe.ErrColumnType) {
		t.Fatalf("source type error = %v, want ErrColumnType", err)
	}
}

func TestTryDerive2PreservesNullsAndAnnotatesErrors(t *testing.T) {
	teams, err := series.NewNullable(
		[]string{"red", "missing", "blue"},
		[]bool{true, false, true},
	)
	if err != nil {
		t.Fatal(err)
	}
	years, err := series.NewNullable(
		[]int{2025, 2025, 2026},
		[]bool{true, true, true},
	)
	if err != nil {
		t.Fatal(err)
	}
	frame, err := dataframe.New().WithSeries("team", teams)
	if err != nil {
		t.Fatal(err)
	}
	frame, err = frame.WithSeries("year", years)
	if err != nil {
		t.Fatal(err)
	}

	calls := 0
	derived, err := frame.TryDerive2("team", "year", "season", func(team string, year int) (seasonKey, error) {
		calls++
		return seasonKey{Team: team, Year: year}, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	seasons, err := derived.Column[seasonKey]("season")
	if err != nil {
		t.Fatal(err)
	}
	if calls != 2 {
		t.Fatalf("callback calls = %d, want 2", calls)
	}
	if got, want := seasons.Validity(), []bool{true, false, true}; !slices.Equal(got, want) {
		t.Fatalf("validity = %v, want %v", got, want)
	}

	boom := errors.New("boom")
	_, err = frame.TryDerive2("team", "year", "season", func(team string, year int) (seasonKey, error) {
		if team == "blue" {
			return seasonKey{}, boom
		}
		return seasonKey{Team: team, Year: year}, nil
	})
	if !errors.Is(err, boom) {
		t.Fatalf("error = %v, want wrapped boom", err)
	}
	if got, want := err.Error(), "derive \"season\" from \"team\" and \"year\": map row 2: boom"; got != want {
		t.Fatalf("error = %q, want %q", got, want)
	}

	dense, err := dataframe.New().WithColumn("left", []int{1, 2})
	if err != nil {
		t.Fatal(err)
	}
	dense, err = dense.WithColumn("right", []int{10, 20})
	if err != nil {
		t.Fatal(err)
	}
	_, err = dense.TryDerive2("left", "right", "sum", func(left, right int) (int, error) {
		if left == 2 {
			return 0, boom
		}
		return left + right, nil
	})
	if err == nil {
		t.Fatal("expected dense error")
	}
	if got, want := err.Error(), "derive \"sum\" from \"left\" and \"right\": map row 1: boom"; got != want {
		t.Fatalf("dense error = %q, want %q", got, want)
	}
}

func TestTryDerivePreservesNullsAndAnnotatesErrors(t *testing.T) {
	numbers, err := series.NewNullable(
		[]int{2, 99, 4},
		[]bool{true, false, true},
	)
	if err != nil {
		t.Fatal(err)
	}
	frame, err := dataframe.New().WithSeries("number", numbers)
	if err != nil {
		t.Fatal(err)
	}

	derived, err := frame.TryDerive("number", "text", func(value int) (string, error) {
		return strconv.Itoa(value), nil
	})
	if err != nil {
		t.Fatal(err)
	}
	texts, err := derived.Column[string]("text")
	if err != nil {
		t.Fatal(err)
	}
	if got, want := texts.Validity(), []bool{true, false, true}; !slices.Equal(got, want) {
		t.Fatalf("validity = %v, want %v", got, want)
	}

	_, err = frame.TryDerive("number", "text", func(value int) (string, error) {
		if value == 4 {
			return "", errors.New("boom")
		}
		return strconv.Itoa(value), nil
	})
	if err == nil {
		t.Fatal("expected error")
	}
	if got, want := err.Error(), "derive \"text\" from \"number\": map row 2: boom"; got != want {
		t.Fatalf("error = %q, want %q", got, want)
	}
}

func TestNullableSchemaDoesNotDependOnCurrentNullCount(t *testing.T) {
	numbers, err := series.NewNullable(
		[]int{2, 4},
		[]bool{true, true},
	)
	if err != nil {
		t.Fatal(err)
	}
	if !numbers.Nullable() || numbers.NullCount() != 0 {
		t.Fatalf("nullable = %v, null count = %d; want true, 0", numbers.Nullable(), numbers.NullCount())
	}

	frame, err := dataframe.New().WithSeries("number", numbers)
	if err != nil {
		t.Fatal(err)
	}
	if schema := frame.Schema(); len(schema) != 1 || !schema[0].Nullable {
		t.Fatalf("schema = %+v, want one nullable field", schema)
	}
}

func TestFrameFillNullReplacesValuesAndRemovesNullableSchema(t *testing.T) {
	ages, err := series.NewNullable(
		[]int{17, 0, 40},
		[]bool{true, false, true},
	)
	if err != nil {
		t.Fatal(err)
	}
	frame, err := dataframe.New().WithSeries("age", ages)
	if err != nil {
		t.Fatal(err)
	}
	frame, err = frame.WithColumn("name", []string{"Ada", "Ben", "Cy"})
	if err != nil {
		t.Fatal(err)
	}

	filled, err := frame.FillNull("age", 25)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := filled.Names(), []string{"age", "name"}; !slices.Equal(got, want) {
		t.Fatalf("names = %v, want %v", got, want)
	}
	filledAges, err := filled.Column[int]("age")
	if err != nil {
		t.Fatal(err)
	}
	if got, want := filledAges.Values(), []int{17, 25, 40}; !slices.Equal(got, want) {
		t.Fatalf("ages = %v, want %v", got, want)
	}
	if filledAges.Nullable() {
		t.Fatal("filled age column is nullable")
	}
	originalAges, err := frame.Column[int]("age")
	if err != nil {
		t.Fatal(err)
	}
	if !originalAges.Nullable() || originalAges.NullCount() != 1 {
		t.Fatalf("source nullable = %v, null count = %d; want true, 1", originalAges.Nullable(), originalAges.NullCount())
	}

	if _, err := frame.FillNull("age", "unknown"); !errors.Is(err, dataframe.ErrColumnType) {
		t.Fatalf("type error = %v, want ErrColumnType", err)
	}
	if _, err := frame.FillNull("missing", 0); !errors.Is(err, dataframe.ErrColumnNotFound) {
		t.Fatalf("missing error = %v, want ErrColumnNotFound", err)
	}

	dense, err := dataframe.New().WithColumn("age", []int{17, 40})
	if err != nil {
		t.Fatal(err)
	}
	dense, err = dense.FillNull("age", 25)
	if err != nil {
		t.Fatal(err)
	}
	denseAges, err := dense.Column[int]("age")
	if err != nil {
		t.Fatal(err)
	}
	if got, want := denseAges.Values(), []int{17, 40}; !slices.Equal(got, want) {
		t.Fatalf("dense ages = %v, want %v", got, want)
	}
	if denseAges.Nullable() {
		t.Fatal("dense filled age column is nullable")
	}
}

func TestFrameDropNullKeepsAlignedRowsAndSchema(t *testing.T) {
	ages, err := series.NewNullable(
		[]int{17, 0, 40, 0},
		[]bool{true, false, true, false},
	)
	if err != nil {
		t.Fatal(err)
	}
	frame, err := dataframe.New().WithSeries("age", ages)
	if err != nil {
		t.Fatal(err)
	}
	frame, err = frame.WithColumn("name", []string{"Ada", "Ben", "Cy", "Dee"})
	if err != nil {
		t.Fatal(err)
	}

	complete, err := frame.DropNull("age")
	if err != nil {
		t.Fatal(err)
	}
	completeAges, err := complete.Column[int]("age")
	if err != nil {
		t.Fatal(err)
	}
	if got, want := completeAges.Values(), []int{17, 40}; !slices.Equal(got, want) {
		t.Fatalf("ages = %v, want %v", got, want)
	}
	if !completeAges.Nullable() || completeAges.NullCount() != 0 {
		t.Fatalf("nullable = %v, null count = %d; want true, 0", completeAges.Nullable(), completeAges.NullCount())
	}
	names, err := complete.Column[string]("name")
	if err != nil {
		t.Fatal(err)
	}
	if got, want := names.Values(), []string{"Ada", "Cy"}; !slices.Equal(got, want) {
		t.Fatalf("names = %v, want %v", got, want)
	}
	if frame.Len() != 4 {
		t.Fatalf("source length = %d, want 4", frame.Len())
	}

	dense, err := dataframe.New().WithColumn("age", []int{17, 40})
	if err != nil {
		t.Fatal(err)
	}
	unchanged, err := dense.DropNull("age")
	if err != nil {
		t.Fatal(err)
	}
	if unchanged.Len() != 2 || unchanged.Width() != 1 {
		t.Fatalf("dense shape = (%d, %d), want (2, 1)", unchanged.Len(), unchanged.Width())
	}

	allPresent, err := series.NewNullable([]int{17, 40}, []bool{true, true})
	if err != nil {
		t.Fatal(err)
	}
	allPresentFrame, err := dataframe.New().WithSeries("age", allPresent)
	if err != nil {
		t.Fatal(err)
	}
	allPresentFrame, err = allPresentFrame.DropNull("age")
	if err != nil {
		t.Fatal(err)
	}
	allPresentAges, err := allPresentFrame.Column[int]("age")
	if err != nil {
		t.Fatal(err)
	}
	if allPresentAges.Len() != 2 || !allPresentAges.Nullable() || allPresentAges.NullCount() != 0 {
		t.Fatalf(
			"all-present result = (len %d, nullable %v, null count %d), want (2, true, 0)",
			allPresentAges.Len(), allPresentAges.Nullable(), allPresentAges.NullCount(),
		)
	}

	allNull, err := series.NewNullable([]int{0, 0}, []bool{false, false})
	if err != nil {
		t.Fatal(err)
	}
	empty, err := dataframe.New().WithSeries("age", allNull)
	if err != nil {
		t.Fatal(err)
	}
	empty, err = empty.WithColumn("name", []string{"Ada", "Ben"})
	if err != nil {
		t.Fatal(err)
	}
	empty, err = empty.DropNull("age")
	if err != nil {
		t.Fatal(err)
	}
	if empty.Len() != 0 || empty.Width() != 2 {
		t.Fatalf("all-null shape = (%d, %d), want (0, 2)", empty.Len(), empty.Width())
	}
	emptyAges, err := empty.Column[int]("age")
	if err != nil {
		t.Fatal(err)
	}
	if !emptyAges.Nullable() {
		t.Fatal("all-null result lost nullable schema")
	}

	if _, err := frame.DropNull("missing"); !errors.Is(err, dataframe.ErrColumnNotFound) {
		t.Fatalf("missing error = %v, want ErrColumnNotFound", err)
	}
}

func TestFrameRejectsWrongTypeAndLength(t *testing.T) {
	frame, err := dataframe.New().WithColumn("age", []int{17, 25})
	if err != nil {
		t.Fatal(err)
	}

	if _, err := frame.Column[int64]("age"); !errors.Is(err, dataframe.ErrColumnType) {
		t.Fatalf("wrong type error = %v", err)
	}
	if _, err := frame.WithColumn("name", []string{"Ada"}); !errors.Is(err, dataframe.ErrRowCount) {
		t.Fatalf("wrong length error = %v", err)
	}
}

func TestFrameAndInputsAreImmutable(t *testing.T) {
	input := []int{1, 2}
	original, err := dataframe.New().WithColumn("number", input)
	if err != nil {
		t.Fatal(err)
	}
	input[0] = 99

	changed, err := original.WithColumn("number", []int{3, 4})
	if err != nil {
		t.Fatal(err)
	}

	originalNumbers, _ := original.Column[int]("number")
	changedNumbers, _ := changed.Column[int]("number")
	if got, want := originalNumbers.Values(), []int{1, 2}; !slices.Equal(got, want) {
		t.Fatalf("original = %v, want %v", got, want)
	}
	if got, want := changedNumbers.Values(), []int{3, 4}; !slices.Equal(got, want) {
		t.Fatalf("changed = %v, want %v", got, want)
	}
}

func TestConcatFramesPreservesOrderAndWidensNullability(t *testing.T) {
	leftScores, err := series.NewNullable(
		[]int{1, 2},
		[]bool{true, true},
	)
	if err != nil {
		t.Fatal(err)
	}
	rightNames, err := series.NewNullable(
		[]string{"Cy", "missing"},
		[]bool{true, false},
	)
	if err != nil {
		t.Fatal(err)
	}

	left, err := dataframe.New().WithColumn("name", []string{"Ada", "Ben"})
	if err != nil {
		t.Fatal(err)
	}
	left, err = left.WithSeries("score", leftScores)
	if err != nil {
		t.Fatal(err)
	}
	right, err := dataframe.New().WithSeries("name", rightNames)
	if err != nil {
		t.Fatal(err)
	}
	right, err = right.WithColumn("score", []int{3, 4})
	if err != nil {
		t.Fatal(err)
	}

	combined, err := left.Concat(right)
	if err != nil {
		t.Fatal(err)
	}
	if combined.Len() != 4 || combined.Width() != 2 {
		t.Fatalf("combined shape = (%d, %d), want (4, 2)", combined.Len(), combined.Width())
	}
	if got, want := combined.Names(), []string{"name", "score"}; !slices.Equal(got, want) {
		t.Fatalf("names = %v, want %v", got, want)
	}

	names, err := combined.Column[string]("name")
	if err != nil {
		t.Fatal(err)
	}
	for row, want := range []string{"Ada", "Ben", "Cy"} {
		if got, valid := names.At(row); !valid || got != want {
			t.Fatalf("name row %d = %q, %v; want %q, true", row, got, valid, want)
		}
	}
	if _, valid := names.At(3); valid {
		t.Fatal("name row 3 is present, want null")
	}
	if !names.Nullable() {
		t.Fatal("name result is not nullable")
	}

	scores, err := combined.Column[int]("score")
	if err != nil {
		t.Fatal(err)
	}
	if got, want := scores.Values(), []int{1, 2, 3, 4}; !slices.Equal(got, want) {
		t.Fatalf("scores = %v, want %v", got, want)
	}
	if got, want := scores.Validity(), []bool{true, true, true, true}; !slices.Equal(got, want) {
		t.Fatalf("score validity = %v, want %v", got, want)
	}
	if !scores.Nullable() {
		t.Fatal("nullable score schema was not preserved")
	}

	if left.Len() != 2 || right.Len() != 2 {
		t.Fatalf("input lengths = %d, %d; want 2, 2", left.Len(), right.Len())
	}
}

func TestConcatNonNullableFramesStayNonNullable(t *testing.T) {
	left, err := dataframe.New().WithColumn("number", []int{1, 2})
	if err != nil {
		t.Fatal(err)
	}
	left, err = left.WithColumn("word", []string{"one", "two"})
	if err != nil {
		t.Fatal(err)
	}
	right, err := dataframe.New().WithColumn("number", []int{3})
	if err != nil {
		t.Fatal(err)
	}
	right, err = right.WithColumn("word", []string{"three"})
	if err != nil {
		t.Fatal(err)
	}

	combined, err := left.Concat(right)
	if err != nil {
		t.Fatal(err)
	}
	for _, field := range combined.Schema() {
		if field.Nullable {
			t.Fatalf("field %q is nullable, want non-nullable", field.Name)
		}
	}
	numbers, err := combined.Column[int]("number")
	if err != nil {
		t.Fatal(err)
	}
	if got, want := numbers.Values(), []int{1, 2, 3}; !slices.Equal(got, want) {
		t.Fatalf("numbers = %v, want %v", got, want)
	}
}

func TestConcatRejectsMismatchedSchemas(t *testing.T) {
	base, err := dataframe.New().WithColumn("number", []int{1})
	if err != nil {
		t.Fatal(err)
	}
	base, err = base.WithColumn("word", []string{"one"})
	if err != nil {
		t.Fatal(err)
	}

	wrongWidth, err := dataframe.New().WithColumn("number", []int{2})
	if err != nil {
		t.Fatal(err)
	}
	wrongName, err := dataframe.New().WithColumn("word", []int{2})
	if err != nil {
		t.Fatal(err)
	}
	wrongName, err = wrongName.WithColumn("number", []string{"two"})
	if err != nil {
		t.Fatal(err)
	}
	wrongType, err := dataframe.New().WithColumn("number", []int64{2})
	if err != nil {
		t.Fatal(err)
	}
	wrongType, err = wrongType.WithColumn("word", []string{"two"})
	if err != nil {
		t.Fatal(err)
	}

	for name, frame := range map[string]dataframe.Frame{
		"width": wrongWidth,
		"name":  wrongName,
		"type":  wrongType,
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := base.Concat(frame); !errors.Is(err, dataframe.ErrSchemaMismatch) {
				t.Fatalf("error = %v, want ErrSchemaMismatch", err)
			}
		})
	}
}

func TestConcatHandlesSingleAndEmptyNullableFrames(t *testing.T) {
	one, err := dataframe.New().WithColumn("number", []int{1})
	if err != nil {
		t.Fatal(err)
	}
	unchanged, err := one.Concat()
	if err != nil {
		t.Fatal(err)
	}
	numbers, err := unchanged.Column[int]("number")
	if err != nil {
		t.Fatal(err)
	}
	if got, want := numbers.Values(), []int{1}; !slices.Equal(got, want) {
		t.Fatalf("single input = %v, want %v", got, want)
	}

	emptyNullable, err := series.NewNullable([]int{}, []bool{})
	if err != nil {
		t.Fatal(err)
	}
	emptyFrame, err := dataframe.New().WithSeries("number", emptyNullable)
	if err != nil {
		t.Fatal(err)
	}
	widened, err := one.Concat(emptyFrame)
	if err != nil {
		t.Fatal(err)
	}
	numbers, err = widened.Column[int]("number")
	if err != nil {
		t.Fatal(err)
	}
	if !numbers.Nullable() {
		t.Fatal("empty nullable Frame did not widen result schema")
	}
	if got, want := numbers.Validity(), []bool{true}; !slices.Equal(got, want) {
		t.Fatalf("widened validity = %v, want %v", got, want)
	}
}

func TestRenamePreservesSchemaPositionAndImmutability(t *testing.T) {
	scores, err := series.NewNullable(
		[]int{10, 20},
		[]bool{true, true},
	)
	if err != nil {
		t.Fatal(err)
	}
	frame, err := dataframe.New().WithColumn("id", []int{1, 2})
	if err != nil {
		t.Fatal(err)
	}
	frame, err = frame.WithSeries("score", scores)
	if err != nil {
		t.Fatal(err)
	}

	renamed, err := frame.Rename("score", "points")
	if err != nil {
		t.Fatal(err)
	}
	wantSchema := []dataframe.Field{
		{Name: "id", Type: reflect.TypeFor[int]()},
		{Name: "points", Type: reflect.TypeFor[int](), Nullable: true},
	}
	if got := renamed.Schema(); !slices.Equal(got, wantSchema) {
		t.Fatalf("schema = %v, want %v", got, wantSchema)
	}
	points, err := renamed.Column[int]("points")
	if err != nil {
		t.Fatal(err)
	}
	if got, want := points.Values(), []int{10, 20}; !slices.Equal(got, want) {
		t.Fatalf("points = %v, want %v", got, want)
	}
	if got, want := frame.Names(), []string{"id", "score"}; !slices.Equal(got, want) {
		t.Fatalf("source names = %v, want %v", got, want)
	}
	if _, err := renamed.Column[int]("score"); !errors.Is(err, dataframe.ErrColumnNotFound) {
		t.Fatalf("old name error = %v, want ErrColumnNotFound", err)
	}

	unchanged, err := frame.Rename("score", "score")
	if err != nil {
		t.Fatal(err)
	}
	if got, want := unchanged.Schema(), frame.Schema(); !slices.Equal(got, want) {
		t.Fatalf("no-op schema = %v, want %v", got, want)
	}
}

func TestRenameReportsNameAndColumnErrors(t *testing.T) {
	frame, err := dataframe.New().WithColumn("id", []int{1})
	if err != nil {
		t.Fatal(err)
	}
	frame, err = frame.WithColumn("score", []int{10})
	if err != nil {
		t.Fatal(err)
	}

	if _, err := frame.Rename("score", ""); !errors.Is(err, dataframe.ErrInvalidName) {
		t.Fatalf("empty name error = %v, want ErrInvalidName", err)
	}
	if _, err := frame.Rename("missing", "points"); !errors.Is(err, dataframe.ErrColumnNotFound) {
		t.Fatalf("missing column error = %v, want ErrColumnNotFound", err)
	}
	if _, err := frame.Rename("score", "id"); !errors.Is(err, dataframe.ErrColumnConflict) {
		t.Fatalf("conflict error = %v, want ErrColumnConflict", err)
	}
}

func TestRenameResolvesJoinConflictWithoutReorderingColumns(t *testing.T) {
	left, err := dataframe.New().WithColumn("id", []int{1})
	if err != nil {
		t.Fatal(err)
	}
	left, err = left.WithColumn("status", []string{"active"})
	if err != nil {
		t.Fatal(err)
	}
	right, err := dataframe.New().WithColumn("id", []int{1})
	if err != nil {
		t.Fatal(err)
	}
	right, err = right.WithColumn("status", []string{"paid"})
	if err != nil {
		t.Fatal(err)
	}
	right, err = right.WithColumn("amount", []int{20})
	if err != nil {
		t.Fatal(err)
	}

	if _, err := left.Join[int](right, "id"); !errors.Is(err, dataframe.ErrColumnConflict) {
		t.Fatalf("join error = %v, want ErrColumnConflict", err)
	}
	renamed, err := right.Rename("status", "order_status")
	if err != nil {
		t.Fatal(err)
	}
	joined, err := left.Join[int](renamed, "id")
	if err != nil {
		t.Fatal(err)
	}
	if got, want := joined.Names(), []string{"id", "status", "order_status", "amount"}; !slices.Equal(got, want) {
		t.Fatalf("joined names = %v, want %v", got, want)
	}
	if got, want := right.Names(), []string{"id", "status", "amount"}; !slices.Equal(got, want) {
		t.Fatalf("source names = %v, want %v", got, want)
	}
}

func TestSlicePreservesFrameSchemaAndRows(t *testing.T) {
	labels, err := series.NewNullable(
		[]string{"one", "missing", "three", "four"},
		[]bool{true, false, true, true},
	)
	if err != nil {
		t.Fatal(err)
	}
	frame, err := dataframe.New().WithColumn("number", []int{1, 2, 3, 4})
	if err != nil {
		t.Fatal(err)
	}
	frame, err = frame.WithSeries("label", labels)
	if err != nil {
		t.Fatal(err)
	}

	sliced := frame.Slice(1, 3)
	if sliced.Len() != 2 || sliced.Width() != 2 {
		t.Fatalf("shape = (%d, %d), want (2, 2)", sliced.Len(), sliced.Width())
	}
	if got, want := sliced.Names(), []string{"number", "label"}; !slices.Equal(got, want) {
		t.Fatalf("names = %v, want %v", got, want)
	}
	numbers, err := sliced.Column[int]("number")
	if err != nil {
		t.Fatal(err)
	}
	if got, want := numbers.Values(), []int{2, 3}; !slices.Equal(got, want) {
		t.Fatalf("numbers = %v, want %v", got, want)
	}
	slicedLabels, err := sliced.Column[string]("label")
	if err != nil {
		t.Fatal(err)
	}
	if got, want := slicedLabels.Validity(), []bool{false, true}; !slices.Equal(got, want) {
		t.Fatalf("label validity = %v, want %v", got, want)
	}

	empty := frame.Slice(2, 2)
	if empty.Len() != 0 || empty.Width() != 2 {
		t.Fatalf("empty shape = (%d, %d), want (0, 2)", empty.Len(), empty.Width())
	}
	emptyLabels, err := empty.Column[string]("label")
	if err != nil {
		t.Fatal(err)
	}
	if !emptyLabels.Nullable() {
		t.Fatal("empty slice lost nullable schema")
	}
	if frame.Len() != 4 {
		t.Fatalf("source length = %d, want 4", frame.Len())
	}
	if zero := dataframe.New().Slice(0, 0); zero.Len() != 0 || zero.Width() != 0 {
		t.Fatalf("zero frame shape = (%d, %d), want (0, 0)", zero.Len(), zero.Width())
	}
}

func TestFrameSlicePanicsForInvalidBounds(t *testing.T) {
	frame, err := dataframe.New().WithColumn("number", []int{1, 2, 3})
	if err != nil {
		t.Fatal(err)
	}
	tests := map[string]struct {
		frame dataframe.Frame
		start int
		end   int
	}{
		"negative start": {frame: frame, start: -1, end: 1},
		"reversed":       {frame: frame, start: 2, end: 1},
		"past end":       {frame: frame, start: 0, end: 4},
		"zero frame":     {frame: dataframe.New(), start: 0, end: 1},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Fatal("Slice did not panic")
				}
			}()
			test.frame.Slice(test.start, test.end)
		})
	}
}

func TestDistinctOnKeepsFirstRowsAndSchema(t *testing.T) {
	teams, err := series.NewNullable(
		[]string{"missing", "blue", "missing", "blue", "red"},
		[]bool{false, true, false, true, true},
	)
	if err != nil {
		t.Fatal(err)
	}
	frame, err := dataframe.New().WithSeries("team", teams)
	if err != nil {
		t.Fatal(err)
	}
	frame, err = frame.WithColumn("score", []int{1, 2, 3, 4, 5})
	if err != nil {
		t.Fatal(err)
	}

	distinct, err := frame.DistinctOn[string]("team")
	if err != nil {
		t.Fatal(err)
	}
	if got, want := distinct.Names(), []string{"team", "score"}; !slices.Equal(got, want) {
		t.Fatalf("names = %v, want %v", got, want)
	}
	distinctTeams, err := distinct.Column[string]("team")
	if err != nil {
		t.Fatal(err)
	}
	if got, want := distinctTeams.Validity(), []bool{false, true, true}; !slices.Equal(got, want) {
		t.Fatalf("team validity = %v, want %v", got, want)
	}
	scores, err := distinct.Column[int]("score")
	if err != nil {
		t.Fatal(err)
	}
	if got, want := scores.Values(), []int{1, 2, 5}; !slices.Equal(got, want) {
		t.Fatalf("scores = %v, want %v", got, want)
	}
	if frame.Len() != 5 {
		t.Fatalf("source length = %d, want 5", frame.Len())
	}
}

func TestDistinctOnPreservesEmptySchema(t *testing.T) {
	frame, err := dataframe.New().WithColumn("team", []string{})
	if err != nil {
		t.Fatal(err)
	}
	frame, err = frame.WithColumn("score", []int{})
	if err != nil {
		t.Fatal(err)
	}

	distinct, err := frame.DistinctOn[string]("team")
	if err != nil {
		t.Fatal(err)
	}
	if distinct.Len() != 0 || distinct.Width() != 2 {
		t.Fatalf("shape = (%d, %d), want (0, 2)", distinct.Len(), distinct.Width())
	}
	if got, want := distinct.Names(), []string{"team", "score"}; !slices.Equal(got, want) {
		t.Fatalf("names = %v, want %v", got, want)
	}
}

func TestDistinctOnReportsTypedColumnErrors(t *testing.T) {
	frame, err := dataframe.New().WithColumn("team", []string{})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := frame.DistinctOn[int]("team"); !errors.Is(err, dataframe.ErrColumnType) {
		t.Fatalf("type error = %v, want ErrColumnType", err)
	}
	if _, err := frame.DistinctOn[string]("missing"); !errors.Is(err, dataframe.ErrColumnNotFound) {
		t.Fatalf("missing error = %v, want ErrColumnNotFound", err)
	}
}

func TestSelectAndDropPreserveFrameInvariants(t *testing.T) {
	frame, err := dataframe.New().WithColumn("number", []int{1, 2})
	if err != nil {
		t.Fatal(err)
	}
	frame, err = frame.WithColumn("word", []string{"one", "two"})
	if err != nil {
		t.Fatal(err)
	}

	selected, err := frame.Select("word", "number")
	if err != nil {
		t.Fatal(err)
	}
	if got, want := selected.Names(), []string{"word", "number"}; !slices.Equal(got, want) {
		t.Fatalf("selected names = %v, want %v", got, want)
	}

	empty, err := frame.Select()
	if err != nil {
		t.Fatal(err)
	}
	if empty.Len() != 0 || empty.Width() != 0 {
		t.Fatalf("empty shape = (%d, %d), want (0, 0)", empty.Len(), empty.Width())
	}

	one, err := frame.Drop("word")
	if err != nil {
		t.Fatal(err)
	}
	empty, err = one.Drop("number")
	if err != nil {
		t.Fatal(err)
	}
	if empty.Len() != 0 || empty.Width() != 0 {
		t.Fatalf("dropped shape = (%d, %d), want (0, 0)", empty.Len(), empty.Width())
	}
}

func TestReplacingSoleColumnMayChangeLength(t *testing.T) {
	frame, err := dataframe.New().WithColumn("number", []int{1, 2})
	if err != nil {
		t.Fatal(err)
	}

	frame, err = frame.WithColumn("number", []int{3, 4, 5})
	if err != nil {
		t.Fatal(err)
	}
	if frame.Len() != 3 {
		t.Fatalf("length = %d, want 3", frame.Len())
	}

	frame, err = frame.WithColumn("word", []string{"three", "four", "five"})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := frame.WithColumn("number", []int{6}); !errors.Is(err, dataframe.ErrRowCount) {
		t.Fatalf("multi-column replacement error = %v, want ErrRowCount", err)
	}
}

func TestFrameOperationsLeaveColumnsDiscoverable(t *testing.T) {
	frame, err := dataframe.New().WithColumn("number", []int{1, 2, 3})
	if err != nil {
		t.Fatal(err)
	}
	frame, err = frame.WithColumn("word", []string{"one", "two", "three"})
	if err != nil {
		t.Fatal(err)
	}

	selected, err := frame.Select("word", "number")
	if err != nil {
		t.Fatal(err)
	}
	selected, err = selected.WithColumn("odd", []bool{true, false, true})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := selected.Column[string]("word"); err != nil {
		t.Fatalf("selected column lookup: %v", err)
	}

	filtered, err := selected.Filter("number", func(number int) bool { return number > 1 })
	if err != nil {
		t.Fatal(err)
	}
	if filtered.Len() != 2 {
		t.Fatalf("filtered length = %d, want 2", filtered.Len())
	}
	if _, err := filtered.Column[bool]("odd"); err != nil {
		t.Fatalf("filtered column lookup: %v", err)
	}
}

func TestLookupsStayCorrectWhenPositionCacheIsReused(t *testing.T) {
	frame, err := dataframe.New().WithColumn("number", []int{1, 2, 3, 4})
	if err != nil {
		t.Fatal(err)
	}
	frame, err = frame.WithColumn("word", []string{"one", "two", "three", "four"})
	if err != nil {
		t.Fatal(err)
	}

	if _, err := frame.Column[int]("number"); err != nil {
		t.Fatal(err)
	}

	sliced := frame.Slice(1, 3)
	numbers, err := sliced.Column[int]("number")
	if err != nil {
		t.Fatalf("sliced column lookup: %v", err)
	}
	if got, want := numbers.Values(), []int{2, 3}; !slices.Equal(got, want) {
		t.Fatalf("sliced values = %v, want %v", got, want)
	}

	filtered, err := frame.Filter("number", func(number int) bool { return number%2 == 0 })
	if err != nil {
		t.Fatal(err)
	}
	words, err := filtered.Column[string]("word")
	if err != nil {
		t.Fatalf("filtered column lookup: %v", err)
	}
	if got, want := words.Values(), []string{"two", "four"}; !slices.Equal(got, want) {
		t.Fatalf("filtered values = %v, want %v", got, want)
	}

	renamed, err := frame.Rename("word", "label")
	if err != nil {
		t.Fatal(err)
	}
	labels, err := renamed.Column[string]("label")
	if err != nil {
		t.Fatalf("renamed column lookup: %v", err)
	}
	if labels.Len() != 4 {
		t.Fatalf("renamed length = %d, want 4", labels.Len())
	}
	if _, err := renamed.Column[string]("word"); !errors.Is(err, dataframe.ErrColumnNotFound) {
		t.Fatalf("old name error = %v, want %v", err, dataframe.ErrColumnNotFound)
	}
}

func TestGroupByAggregatesInFirstSeenOrder(t *testing.T) {
	scores, err := series.NewNullable(
		[]int{10, 20, 99, 40, 30},
		[]bool{true, true, false, false, true},
	)
	if err != nil {
		t.Fatal(err)
	}

	frame, err := dataframe.New().WithColumn("team", []string{"blue", "red", "blue", "green", "red"})
	if err != nil {
		t.Fatal(err)
	}
	frame, err = frame.WithSeries("score", scores)
	if err != nil {
		t.Fatal(err)
	}

	grouped, err := frame.GroupBy[string]("team")
	if err != nil {
		t.Fatal(err)
	}
	withTotals, err := grouped.WithAggregate[int]("score", "total", series.Sum)
	if err != nil {
		t.Fatal(err)
	}
	withAverages, err := withTotals.WithAggregate[int]("score", "average", series.Mean)
	if err != nil {
		t.Fatal(err)
	}

	summary := withAverages.Frame()
	if got, want := summary.Names(), []string{"team", "total", "average"}; !slices.Equal(got, want) {
		t.Fatalf("names = %v, want %v", got, want)
	}

	teams, err := summary.Column[string]("team")
	if err != nil {
		t.Fatal(err)
	}
	if got, want := teams.Values(), []string{"blue", "red", "green"}; !slices.Equal(got, want) {
		t.Fatalf("teams = %v, want %v", got, want)
	}

	totals, err := summary.Column[int]("total")
	if err != nil {
		t.Fatal(err)
	}
	if got, want := totals.Values(), []int{10, 50, 0}; !slices.Equal(got, want) {
		t.Fatalf("totals = %v, want %v", got, want)
	}
	if got, want := totals.Validity(), []bool{true, true, false}; !slices.Equal(got, want) {
		t.Fatalf("total validity = %v, want %v", got, want)
	}
	if !totals.Nullable() {
		t.Fatal("aggregate result is not nullable")
	}

	averages, err := summary.Column[float64]("average")
	if err != nil {
		t.Fatal(err)
	}
	if got, want := averages.Values(), []float64{10, 25, 0}; !slices.Equal(got, want) {
		t.Fatalf("averages = %v, want %v", got, want)
	}
	if got, want := averages.Validity(), []bool{true, true, false}; !slices.Equal(got, want) {
		t.Fatalf("average validity = %v, want %v", got, want)
	}

	if got, want := withTotals.Frame().Names(), []string{"team", "total"}; !slices.Equal(got, want) {
		t.Fatalf("previous grouped names = %v, want %v", got, want)
	}
	if frame.Len() != 5 || frame.Width() != 2 {
		t.Fatalf("source shape = (%d, %d), want (5, 2)", frame.Len(), frame.Width())
	}
}

func TestGroupByCombinesNullKeys(t *testing.T) {
	teams, err := series.NewNullable(
		[]string{"missing", "blue", "missing", "blue", "red"},
		[]bool{false, true, false, true, true},
	)
	if err != nil {
		t.Fatal(err)
	}

	frame, err := dataframe.New().WithSeries("team", teams)
	if err != nil {
		t.Fatal(err)
	}
	frame, err = frame.WithColumn("score", []int{1, 2, 3, 4, 5})
	if err != nil {
		t.Fatal(err)
	}

	grouped, err := frame.GroupBy[string]("team")
	if err != nil {
		t.Fatal(err)
	}
	grouped, err = grouped.WithAggregate[int]("score", "total", series.Sum)
	if err != nil {
		t.Fatal(err)
	}

	summary := grouped.Frame()
	groupedTeams, err := summary.Column[string]("team")
	if err != nil {
		t.Fatal(err)
	}
	if got, want := groupedTeams.Validity(), []bool{false, true, true}; !slices.Equal(got, want) {
		t.Fatalf("team validity = %v, want %v", got, want)
	}
	if _, valid := groupedTeams.At(0); valid {
		t.Fatal("null group key is present")
	}
	if got, valid := groupedTeams.At(1); !valid || got != "blue" {
		t.Fatalf("second key = %q, %v; want blue, true", got, valid)
	}
	if got, valid := groupedTeams.At(2); !valid || got != "red" {
		t.Fatalf("third key = %q, %v; want red, true", got, valid)
	}

	totals, err := summary.Column[int]("total")
	if err != nil {
		t.Fatal(err)
	}
	if got, want := totals.Values(), []int{4, 6, 5}; !slices.Equal(got, want) {
		t.Fatalf("totals = %v, want %v", got, want)
	}
}

func TestGroupedWithAggregateReplacesResultInPlace(t *testing.T) {
	frame, err := dataframe.New().WithColumn("team", []string{"blue", "blue", "red"})
	if err != nil {
		t.Fatal(err)
	}
	frame, err = frame.WithColumn("score", []int{2, 3, 8})
	if err != nil {
		t.Fatal(err)
	}

	grouped, err := frame.GroupBy[string]("team")
	if err != nil {
		t.Fatal(err)
	}
	grouped, err = grouped.WithAggregate[int]("score", "value", series.Sum)
	if err != nil {
		t.Fatal(err)
	}
	replaced, err := grouped.WithAggregate[int]("score", "value", series.Mean)
	if err != nil {
		t.Fatal(err)
	}

	if got, want := replaced.Frame().Names(), []string{"team", "value"}; !slices.Equal(got, want) {
		t.Fatalf("names = %v, want %v", got, want)
	}
	values, err := replaced.Frame().Column[float64]("value")
	if err != nil {
		t.Fatal(err)
	}
	if got, want := values.Values(), []float64{2.5, 8}; !slices.Equal(got, want) {
		t.Fatalf("values = %v, want %v", got, want)
	}
	if _, err := grouped.Frame().Column[int]("value"); err != nil {
		t.Fatalf("previous aggregate changed: %v", err)
	}
}

func TestGroupByReportsTypedColumnErrors(t *testing.T) {
	frame, err := dataframe.New().WithColumn("team", []string{"blue", "red"})
	if err != nil {
		t.Fatal(err)
	}
	frame, err = frame.WithColumn("score", []int{2, 3})
	if err != nil {
		t.Fatal(err)
	}

	if _, err := frame.GroupBy[int]("team"); !errors.Is(err, dataframe.ErrColumnType) {
		t.Fatalf("key type error = %v, want ErrColumnType", err)
	}
	if _, err := frame.GroupBy[string]("missing"); !errors.Is(err, dataframe.ErrColumnNotFound) {
		t.Fatalf("missing key error = %v, want ErrColumnNotFound", err)
	}

	grouped, err := frame.GroupBy[string]("team")
	if err != nil {
		t.Fatal(err)
	}
	wrongType := func(series.Series[string]) (int, bool) {
		return 0, true
	}
	if _, err := grouped.WithAggregate("score", "total", wrongType); !errors.Is(err, dataframe.ErrColumnType) {
		t.Fatalf("aggregate type error = %v, want ErrColumnType", err)
	}
	if _, err := grouped.WithAggregate[int]("missing", "total", series.Sum); !errors.Is(err, dataframe.ErrColumnNotFound) {
		t.Fatalf("missing aggregate column error = %v, want ErrColumnNotFound", err)
	}
	if _, err := grouped.WithAggregate[int]("score", "team", series.Sum); !errors.Is(err, dataframe.ErrGroupKey) {
		t.Fatalf("group key error = %v, want ErrGroupKey", err)
	}

	called := false
	invalidName := func(series.Series[int]) (int, bool) {
		called = true
		return 0, true
	}
	if _, err := grouped.WithAggregate("score", "", invalidName); !errors.Is(err, dataframe.ErrInvalidName) {
		t.Fatalf("result name error = %v, want ErrInvalidName", err)
	}
	if called {
		t.Fatal("aggregate called for an invalid result name")
	}
}

func TestGroupByEmptyFrameColumn(t *testing.T) {
	frame, err := dataframe.New().WithColumn("team", []string{})
	if err != nil {
		t.Fatal(err)
	}
	frame, err = frame.WithColumn("score", []int{})
	if err != nil {
		t.Fatal(err)
	}

	grouped, err := frame.GroupBy[string]("team")
	if err != nil {
		t.Fatal(err)
	}
	called := false
	grouped, err = grouped.WithAggregate("score", "total", func(series.Series[int]) (int, bool) {
		called = true
		return 0, true
	})
	if err != nil {
		t.Fatal(err)
	}
	if called {
		t.Fatal("aggregate called for an empty grouping")
	}
	if result := grouped.Frame(); result.Len() != 0 || result.Width() != 2 {
		t.Fatalf("result shape = (%d, %d), want (0, 2)", result.Len(), result.Width())
	}
	totals, err := grouped.Frame().Column[int]("total")
	if err != nil {
		t.Fatal(err)
	}
	if !totals.Nullable() {
		t.Fatal("empty aggregate result is not nullable")
	}
}

func TestSortIsStableAndImmutable(t *testing.T) {
	frame := dataframe.New()
	var err error
	frame, err = frame.WithColumn("name", []string{"first", "second", "third", "fourth"})
	if err != nil {
		t.Fatal(err)
	}
	frame, err = frame.WithColumn("score", []int{2, 1, 2, 1})
	if err != nil {
		t.Fatal(err)
	}

	sorted, err := frame.Sort[int]("score", dataframe.SortOptions{})
	if err != nil {
		t.Fatal(err)
	}

	names, err := sorted.Column[string]("name")
	if err != nil {
		t.Fatal(err)
	}
	if got, want := names.Values(), []string{"second", "fourth", "first", "third"}; !slices.Equal(got, want) {
		t.Fatalf("sorted names = %v, want %v", got, want)
	}

	originalNames, err := frame.Column[string]("name")
	if err != nil {
		t.Fatal(err)
	}
	if got, want := originalNames.Values(), []string{"first", "second", "third", "fourth"}; !slices.Equal(got, want) {
		t.Fatalf("original names = %v, want %v", got, want)
	}
}

func TestSortDescendingWithNullsFirst(t *testing.T) {
	scores, err := series.NewNullable(
		[]int{2, 0, 3, 1},
		[]bool{true, false, true, true},
	)
	if err != nil {
		t.Fatal(err)
	}

	frame := dataframe.New()
	frame, err = frame.WithColumn("name", []string{"two", "missing", "three", "one"})
	if err != nil {
		t.Fatal(err)
	}
	frame, err = frame.WithSeries("score", scores)
	if err != nil {
		t.Fatal(err)
	}

	sorted, err := frame.Sort[int]("score", dataframe.SortOptions{
		Descending: true,
		NullsFirst: true,
	})
	if err != nil {
		t.Fatal(err)
	}

	names, err := sorted.Column[string]("name")
	if err != nil {
		t.Fatal(err)
	}
	if got, want := names.Values(), []string{"missing", "three", "two", "one"}; !slices.Equal(got, want) {
		t.Fatalf("sorted names = %v, want %v", got, want)
	}
}

func TestSortByInfersType(t *testing.T) {
	frame, err := dataframe.New().WithColumn("name", []string{"three", "one", "six", "twelve"})
	if err != nil {
		t.Fatal(err)
	}

	sorted, err := frame.SortBy("name", func(left, right string) int {
		return cmp.Compare(len(left), len(right))
	}, dataframe.SortOptions{})
	if err != nil {
		t.Fatal(err)
	}

	names, err := sorted.Column[string]("name")
	if err != nil {
		t.Fatal(err)
	}
	if got, want := names.Values(), []string{"one", "six", "three", "twelve"}; !slices.Equal(got, want) {
		t.Fatalf("sorted names = %v, want %v", got, want)
	}
}

func TestSortRejectsWrongColumnType(t *testing.T) {
	frame, err := dataframe.New().WithColumn("score", []int{2, 1})
	if err != nil {
		t.Fatal(err)
	}

	if _, err := frame.Sort[int64]("score", dataframe.SortOptions{}); !errors.Is(err, dataframe.ErrColumnType) {
		t.Fatalf("column type error = %v, want ErrColumnType", err)
	}
}

func TestJoinPreservesStableCartesianOrder(t *testing.T) {
	left := joinWithColumn(t, dataframe.New(), "id", []int{2, 1, 2, 3})
	left = joinWithColumn(t, left, "customer", []string{"left 2a", "left 1", "left 2b", "left 3"})
	right := joinWithColumn(t, dataframe.New(), "id", []int{2, 2, 1, 4})
	right = joinWithColumn(t, right, "order", []string{"right 2a", "right 2b", "right 1", "right 4"})

	joined, err := left.Join[int](right, "id")
	if err != nil {
		t.Fatal(err)
	}
	if got, want := joined.Names(), []string{"id", "customer", "order"}; !slices.Equal(got, want) {
		t.Fatalf("names = %v, want %v", got, want)
	}

	ids, err := joined.Column[int]("id")
	if err != nil {
		t.Fatal(err)
	}
	if got, want := ids.Values(), []int{2, 2, 1, 2, 2}; !slices.Equal(got, want) {
		t.Fatalf("ids = %v, want %v", got, want)
	}
	customers, err := joined.Column[string]("customer")
	if err != nil {
		t.Fatal(err)
	}
	if got, want := customers.Values(), []string{"left 2a", "left 2a", "left 1", "left 2b", "left 2b"}; !slices.Equal(got, want) {
		t.Fatalf("customers = %v, want %v", got, want)
	}
	orders, err := joined.Column[string]("order")
	if err != nil {
		t.Fatal(err)
	}
	if got, want := orders.Values(), []string{"right 2a", "right 2b", "right 1", "right 2a", "right 2b"}; !slices.Equal(got, want) {
		t.Fatalf("orders = %v, want %v", got, want)
	}

	if left.Len() != 4 || right.Len() != 4 {
		t.Fatalf("input lengths = %d, %d; want 4, 4", left.Len(), right.Len())
	}
}

func TestJoinOnSkipsNullKeysAndPreservesSchema(t *testing.T) {
	leftIDs, err := series.NewNullable(
		[]int{1, 0, 2},
		[]bool{true, false, true},
	)
	if err != nil {
		t.Fatal(err)
	}
	rightIDs, err := series.NewNullable(
		[]int{2, 0, 1},
		[]bool{true, false, true},
	)
	if err != nil {
		t.Fatal(err)
	}

	left := joinWithSeries(t, dataframe.New(), "id", leftIDs)
	left = joinWithColumn(t, left, "name", []string{"one", "missing", "two"})
	right := joinWithSeries(t, dataframe.New(), "customer_id", rightIDs)
	right = joinWithColumn(t, right, "amount", []int{20, 99, 10})

	joined, err := left.JoinOn[int](right, "id", "customer_id")
	if err != nil {
		t.Fatal(err)
	}
	if got, want := joined.Names(), []string{"id", "name", "amount"}; !slices.Equal(got, want) {
		t.Fatalf("names = %v, want %v", got, want)
	}
	ids, err := joined.Column[int]("id")
	if err != nil {
		t.Fatal(err)
	}
	if got, want := ids.Values(), []int{1, 2}; !slices.Equal(got, want) {
		t.Fatalf("ids = %v, want %v", got, want)
	}
	if !ids.Nullable() {
		t.Fatal("nullable left key schema was not preserved")
	}
	amounts, err := joined.Column[int]("amount")
	if err != nil {
		t.Fatal(err)
	}
	if got, want := amounts.Values(), []int{10, 20}; !slices.Equal(got, want) {
		t.Fatalf("amounts = %v, want %v", got, want)
	}
	if amounts.Nullable() {
		t.Fatal("non-nullable right column became nullable")
	}
}

func TestJoinReportsColumnAndKeyErrors(t *testing.T) {
	left := joinWithColumn(t, dataframe.New(), "id", []int{1})
	left = joinWithColumn(t, left, "name", []string{"left"})
	conflicting := joinWithColumn(t, dataframe.New(), "id", []int{1})
	conflicting = joinWithColumn(t, conflicting, "name", []string{"right"})
	if _, err := left.Join[int](conflicting, "id"); !errors.Is(err, dataframe.ErrColumnConflict) {
		t.Fatalf("conflict error = %v, want ErrColumnConflict", err)
	}
	if _, err := left.LeftJoin[int](conflicting, "id"); !errors.Is(err, dataframe.ErrColumnConflict) {
		t.Fatalf("left join conflict error = %v, want ErrColumnConflict", err)
	}

	wrongType := joinWithColumn(t, dataframe.New(), "id", []int64{1})
	if _, err := left.Join[int](wrongType, "id"); !errors.Is(err, dataframe.ErrColumnType) {
		t.Fatalf("right key type error = %v, want ErrColumnType", err)
	}
	if _, err := left.Join[int64](wrongType, "id"); !errors.Is(err, dataframe.ErrColumnType) {
		t.Fatalf("left key type error = %v, want ErrColumnType", err)
	}

	missing := joinWithColumn(t, dataframe.New(), "value", []int{1})
	if _, err := left.Join[int](missing, "id"); !errors.Is(err, dataframe.ErrColumnNotFound) {
		t.Fatalf("missing right key error = %v, want ErrColumnNotFound", err)
	}
	if _, err := missing.Join[int](left, "id"); !errors.Is(err, dataframe.ErrColumnNotFound) {
		t.Fatalf("missing left key error = %v, want ErrColumnNotFound", err)
	}
}

func TestJoinWithNoMatchesPreservesOutputSchema(t *testing.T) {
	newLeft := func(ids []int, names []string) dataframe.Frame {
		frame := joinWithColumn(t, dataframe.New(), "id", ids)
		return joinWithColumn(t, frame, "name", names)
	}
	newRight := func(ids, amounts []int) dataframe.Frame {
		frame := joinWithColumn(t, dataframe.New(), "id", ids)
		validity := make([]bool, len(amounts))
		for i := range validity {
			validity[i] = true
		}
		return joinWithNullableColumn(t, frame, "amount", amounts, validity)
	}

	wantSchema := []dataframe.Field{
		{Name: "id", Type: reflect.TypeFor[int]()},
		{Name: "name", Type: reflect.TypeFor[string]()},
		{Name: "amount", Type: reflect.TypeFor[int](), Nullable: true},
	}
	tests := map[string]struct {
		left  dataframe.Frame
		right dataframe.Frame
	}{
		"no matches":  {left: newLeft([]int{1}, []string{"one"}), right: newRight([]int{2}, []int{20})},
		"empty left":  {left: newLeft([]int{}, []string{}), right: newRight([]int{2}, []int{20})},
		"empty right": {left: newLeft([]int{1}, []string{"one"}), right: newRight([]int{}, []int{})},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			joined, err := test.left.Join[int](test.right, "id")
			if err != nil {
				t.Fatal(err)
			}
			if joined.Len() != 0 {
				t.Fatalf("length = %d, want 0", joined.Len())
			}
			if got := joined.Schema(); !slices.Equal(got, wantSchema) {
				t.Fatalf("schema = %v, want %v", got, wantSchema)
			}
		})
	}
}

func TestLeftJoinPreservesStableRowsAndNullExtendsRightColumns(t *testing.T) {
	left := joinWithColumn(t, dataframe.New(), "id", []int{2, 1, 2, 3})
	left = joinWithColumn(t, left, "customer", []string{"left 2a", "left 1", "left 2b", "left 3"})
	right := joinWithColumn(t, dataframe.New(), "id", []int{2, 2, 4})
	right = joinWithNullableColumn(
		t,
		right,
		"amount",
		[]int{20, 21, 40},
		[]bool{true, false, true},
	)

	joined, err := left.LeftJoin[int](right, "id")
	if err != nil {
		t.Fatal(err)
	}
	if got, want := joined.Names(), []string{"id", "customer", "amount"}; !slices.Equal(got, want) {
		t.Fatalf("names = %v, want %v", got, want)
	}
	ids, err := joined.Column[int]("id")
	if err != nil {
		t.Fatal(err)
	}
	if got, want := ids.Values(), []int{2, 2, 1, 2, 2, 3}; !slices.Equal(got, want) {
		t.Fatalf("ids = %v, want %v", got, want)
	}
	customers, err := joined.Column[string]("customer")
	if err != nil {
		t.Fatal(err)
	}
	if got, want := customers.Values(), []string{"left 2a", "left 2a", "left 1", "left 2b", "left 2b", "left 3"}; !slices.Equal(got, want) {
		t.Fatalf("customers = %v, want %v", got, want)
	}
	amounts, err := joined.Column[int]("amount")
	if err != nil {
		t.Fatal(err)
	}
	if got, want := amounts.Validity(), []bool{true, false, false, true, false, false}; !slices.Equal(got, want) {
		t.Fatalf("amount validity = %v, want %v", got, want)
	}
	if first, valid := amounts.At(0); first != 20 || !valid {
		t.Fatalf("first amount = (%d, %v), want (20, true)", first, valid)
	}
	if second, valid := amounts.At(3); second != 20 || !valid {
		t.Fatalf("second amount = (%d, %v), want (20, true)", second, valid)
	}
	if left.Len() != 4 || right.Len() != 3 {
		t.Fatalf("input lengths = %d, %d; want 4, 3", left.Len(), right.Len())
	}
}

func TestLeftJoinOnOmitsRightKeyAndWidensRightSchema(t *testing.T) {
	left := joinWithColumn(t, dataframe.New(), "id", []int{1, 2})
	right := joinWithColumn(t, dataframe.New(), "customer_id", []int{2, 1})
	right = joinWithColumn(t, right, "amount", []int{20, 10})

	joined, err := left.LeftJoinOn[int](right, "id", "customer_id")
	if err != nil {
		t.Fatal(err)
	}
	if got, want := joined.Names(), []string{"id", "amount"}; !slices.Equal(got, want) {
		t.Fatalf("names = %v, want %v", got, want)
	}
	amounts, err := joined.Column[int]("amount")
	if err != nil {
		t.Fatal(err)
	}
	if got, want := amounts.Values(), []int{10, 20}; !slices.Equal(got, want) {
		t.Fatalf("amounts = %v, want %v", got, want)
	}
	if !amounts.Nullable() {
		t.Fatal("left join did not make a fully matched right column nullable")
	}
	if got, want := amounts.Validity(), []bool{true, true}; !slices.Equal(got, want) {
		t.Fatalf("amount validity = %v, want %v", got, want)
	}
}

func TestLeftJoinWithEmptyRightKeepsLeftRowsAndOutputSchema(t *testing.T) {
	left := joinWithColumn(t, dataframe.New(), "id", []int{1, 2})
	left = joinWithColumn(t, left, "name", []string{"one", "two"})
	right := joinWithColumn(t, dataframe.New(), "id", []int{})
	right = joinWithColumn(t, right, "amount", []int{})

	joined, err := left.LeftJoin[int](right, "id")
	if err != nil {
		t.Fatal(err)
	}
	if got, want := joined.Names(), []string{"id", "name", "amount"}; !slices.Equal(got, want) {
		t.Fatalf("names = %v, want %v", got, want)
	}
	if joined.Len() != 2 {
		t.Fatalf("length = %d, want 2", joined.Len())
	}
	amounts, err := joined.Column[int]("amount")
	if err != nil {
		t.Fatal(err)
	}
	if got, want := amounts.Validity(), []bool{false, false}; !slices.Equal(got, want) {
		t.Fatalf("amount validity = %v, want %v", got, want)
	}

	emptyLeft := joinWithColumn(t, dataframe.New(), "id", []int{})
	emptyLeft = joinWithColumn(t, emptyLeft, "name", []string{})
	nonEmptyRight := joinWithColumn(t, dataframe.New(), "id", []int{1})
	nonEmptyRight = joinWithColumn(t, nonEmptyRight, "amount", []int{10})
	emptyJoined, err := emptyLeft.LeftJoin[int](nonEmptyRight, "id")
	if err != nil {
		t.Fatal(err)
	}
	if emptyJoined.Len() != 0 {
		t.Fatalf("empty left join length = %d, want 0", emptyJoined.Len())
	}
	emptyAmounts, err := emptyJoined.Column[int]("amount")
	if err != nil {
		t.Fatal(err)
	}
	if !emptyAmounts.Nullable() {
		t.Fatal("empty left join right column is non-nullable")
	}
}

func joinWithColumn[T any](t *testing.T, frame dataframe.Frame, name string, values []T) dataframe.Frame {
	t.Helper()
	result, err := frame.WithColumn(name, values)
	if err != nil {
		t.Fatal(err)
	}
	return result
}

func joinWithNullableColumn[T any](
	t *testing.T,
	frame dataframe.Frame,
	name string,
	values []T,
	validity []bool,
) dataframe.Frame {
	t.Helper()
	result, err := frame.WithNullableColumn(name, values, validity)
	if err != nil {
		t.Fatal(err)
	}
	return result
}

func joinWithSeries[T any](t *testing.T, frame dataframe.Frame, name string, values series.Series[T]) dataframe.Frame {
	t.Helper()
	result, err := frame.WithSeries(name, values)
	if err != nil {
		t.Fatal(err)
	}
	return result
}
