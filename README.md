# dataframe

[![Go Reference](https://pkg.go.dev/badge/github.com/joeychilson/dataframe.svg)](https://pkg.go.dev/github.com/joeychilson/dataframe)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A small, immutable, column-oriented dataframe library built for Go 1.27 generic
methods.

> [!NOTE]
> This project is experimental and its API may change while the core feature set
> is developed.

## Requirements

- Go 1.27 or newer

## Install

```sh
go get github.com/joeychilson/dataframe
```

## Example

```go
package main

import (
	"fmt"
	"log"

	"github.com/joeychilson/dataframe"
)

func main() {
	frame := dataframe.New()

	frame, err := frame.WithColumn(
		"name",
		[]string{"Ada", "Ben", "Cy", "Dee"},
	)
	if err != nil {
		log.Fatal(err)
	}

	frame, err = frame.WithNullableColumn(
		"score",
		[]int{80, 0, 95, 65},
		[]bool{true, false, true, true},
	)
	if err != nil {
		log.Fatal(err)
	}

	frame, err = frame.Derive("score", "passed", func(score int) bool {
		return score >= 70
	})
	if err != nil {
		log.Fatal(err)
	}

	passing, err := frame.Filter("passed", func(passed bool) bool {
		return passed
	})
	if err != nil {
		log.Fatal(err)
	}

	passing, err = passing.Sort[int](
		"score",
		dataframe.SortOptions{Descending: true},
	)
	if err != nil {
		log.Fatal(err)
	}

	names, err := passing.Column[string]("name")
	if err != nil {
		log.Fatal(err)
	}
	scores, err := passing.Column[int]("score")
	if err != nil {
		log.Fatal(err)
	}

	for row, score := range scores.Each() {
		name, _ := names.At(row)
		fmt.Printf("%s: %d\n", name, score)
	}

	// Output:
	// Cy: 95
	// Ada: 80
}
```

## License

[MIT](LICENSE) © Joey Chilson
