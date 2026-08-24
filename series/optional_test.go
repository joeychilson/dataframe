package series

import "testing"

func TestSome_ConstructsPresentValue(t *testing.T) {
	optional := Some(42)

	if optional.Value != 42 || !optional.Valid {
		t.Fatalf("Some(42) = %+v, want {Value:42 Valid:true}", optional)
	}
	if value, ok := optional.Get(); value != 42 || !ok {
		t.Fatalf("Some(42).Get() = (%d, %t), want (42, true)", value, ok)
	}
	if value := optional.Or(7); value != 42 {
		t.Fatalf("Some(42).Or(7) = %d, want 42", value)
	}
}

func TestNone_ConstructsAbsentValue(t *testing.T) {
	tests := []struct {
		name     string
		optional Optional[int]
	}{
		{name: "None constructs an absent value", optional: None[int]()},
		{name: "the zero value is absent", optional: Optional[int]{}},
		{name: "an invalid optional ignores its stored value", optional: Optional[int]{Value: 42}},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if value, ok := test.optional.Get(); value != 0 || ok {
				t.Fatalf("Get() = (%d, %t), want (0, false)", value, ok)
			}
			if value := test.optional.Or(7); value != 7 {
				t.Fatalf("Or(7) = %d, want 7", value)
			}
		})
	}
}
