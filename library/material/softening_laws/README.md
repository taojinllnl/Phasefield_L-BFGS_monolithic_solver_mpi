# Softening-law profiles

Each `*.softening` file defines exactly three numeric key-value pairs:

```text
p = 2.0;
a2 = -0.5;
a3 = 0.0;
```

Select a file in the `Materials` PRM subsection by its name without the
extension:

```text
set Softening laws = 0: PFCZM_Linear
```

Adding another valid `*.softening` file makes the new law available without a
C++ source change. Standard AT1 and AT2 cases should omit `Softening laws`.
