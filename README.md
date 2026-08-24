# dataframe

[![CI](https://github.com/joeychilson/dataframe/actions/workflows/ci.yml/badge.svg)](https://github.com/joeychilson/dataframe/actions/workflows/ci.yml)
[![Go Reference](https://pkg.go.dev/badge/github.com/joeychilson/dataframe.svg)](https://pkg.go.dev/github.com/joeychilson/dataframe)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An immutable, column-oriented dataframe library.

Requires Go 1.27 or later.

## Installation

```text
go get github.com/joeychilson/dataframe
```

## Packages

- `dataframe` — immutable frames, sorting, grouping, joins, and records.
- `series` — typed nullable columns, transforms, kernels, and reductions.
- `mask` — positional row selections constructed with `mask.New`, `mask.All`, or `mask.None`.
- `csv` — CSV frame and record I/O.
- `sql` — `database/sql` frame and record I/O through queries and prepared statements.

## Example

Build a revenue report from ordinary Go records. The example filters paid
orders, joins customer data, groups the result by region, and sorts the final
report by revenue.

```go
package main

import (
	"fmt"
	"log"

	"github.com/joeychilson/dataframe"
	"github.com/joeychilson/dataframe/series"
)

type Customer struct {
	ID     int    `df:"customer_id"`
	Region string `df:"region"`
}

type Order struct {
	ID         int     `df:"order_id"`
	CustomerID int     `df:"customer_id"`
	Status     string  `df:"status"`
	Total      float64 `df:"total"`
}

type RegionSummary struct {
	Region  string  `df:"region"`
	Orders  int     `df:"orders"`
	Revenue float64 `df:"revenue"`
}

func main() {
	customers, err := dataframe.FromRecords([]Customer{
		{ID: 101, Region: "North"},
		{ID: 102, Region: "South"},
		{ID: 103, Region: "North"},
		{ID: 104, Region: "West"},
	})
	if err != nil {
		log.Fatal(err)
	}

	orders, err := dataframe.FromRecords([]Order{
		{ID: 1001, CustomerID: 101, Status: "paid", Total: 120.50},
		{ID: 1002, CustomerID: 102, Status: "paid", Total: 80.00},
		{ID: 1003, CustomerID: 101, Status: "refunded", Total: 75.00},
		{ID: 1004, CustomerID: 103, Status: "paid", Total: 210.00},
		{ID: 1005, CustomerID: 104, Status: "paid", Total: 150.00},
		{ID: 1006, CustomerID: 102, Status: "paid", Total: 45.00},
	})
	if err != nil {
		log.Fatal(err)
	}

	status, err := orders.Column[string]("status")
	if err != nil {
		log.Fatal(err)
	}
	paid := orders.Filter(series.EqualValue(status, "paid"))

	paid, err = paid.InnerJoin(customers, dataframe.Using[int]("customer_id"))
	if err != nil {
		log.Fatal(err)
	}

	regions, err := paid.Column[string]("region")
	if err != nil {
		log.Fatal(err)
	}
	totals, err := paid.Column[float64]("total")
	if err != nil {
		log.Fatal(err)
	}

	byRegion := paid.GroupBy(regions)
	report, err := byRegion.Result(
		"region",
		dataframe.ColumnFromSeries("orders", byRegion.Count(totals)),
		dataframe.ColumnFromSeries("revenue", byRegion.Sum(totals)),
	)
	if err != nil {
		log.Fatal(err)
	}

	revenue, err := report.Column[float64]("revenue")
	if err != nil {
		log.Fatal(err)
	}
	report = report.SortedBy(dataframe.Desc(revenue))

	summaries, err := report.Records[RegionSummary]()
	if err != nil {
		log.Fatal(err)
	}
	for _, summary := range summaries {
		fmt.Printf("%s: orders=%d revenue=$%.2f\n", summary.Region, summary.Orders, summary.Revenue)
	}

	// Output:
	// North: orders=2 revenue=$330.50
	// West: orders=1 revenue=$150.00
	// South: orders=2 revenue=$125.00
}
```

## Development

```text
golangci-lint run ./...
go test ./...
go test -race ./...
go build ./...
```

## Release

Push a semantic version tag in the form `vMAJOR.MINOR.PATCH`, optionally with a
prerelease suffix, to create a GitHub Release with generated notes. The release
publishes the Go module source and does not include binary artifacts.

## License

[MIT](LICENSE)
