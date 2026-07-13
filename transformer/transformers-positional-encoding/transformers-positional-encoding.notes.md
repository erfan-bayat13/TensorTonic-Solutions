## What are we actually trying to build?

Suppose:

```python
max_len = 4
d_model = 6
```

We need a positional encoding table with **4 positions** and **6 embedding dimensions**:

```text
             embedding dimension
             0    1    2    3    4    5

position 0   ?    ?    ?    ?    ?    ?
position 1   ?    ?    ?    ?    ?    ?
position 2   ?    ?    ?    ?    ?    ?
position 3   ?    ?    ?    ?    ?    ?

```

So we start with:

```python
pe = torch.zeros(max_len, d_model)
```

which is:

```text
[
  [0, 0, 0, 0, 0, 0],   ← position 0
  [0, 0, 0, 0, 0, 0],   ← position 1
  [0, 0, 0, 0, 0, 0],   ← position 2
  [0, 0, 0, 0, 0, 0]    ← position 3
]
```

---
## Step 1: Separate even and odd columns

The rule says:

```text
even dimensions → sin
odd dimensions  → cos
```

So:

```text
dimension:     0      1      2      3      4      5
              sin    cos    sin    cos    sin    cos

```

Python slicing gives us exactly those columns:

```python
pe[:, 0::2]

```

means:

```text
all rows
   ↓
   :

start at column 0
↓
0 :: 2
     ↑
  jump by 2

```

So:

```python
pe[:, 0::2]

```

selects:

```text
columns 0, 2, 4

```

And:

```python
pe[:, 1::2]

```

selects:

```text
columns 1, 3, 5

```

Visually:

```text
             0    1    2    3    4    5
             ↓         ↓         ↓
pe[:,0::2]   X         X         X

                  ↓         ↓         ↓
pe[:,1::2]        X         X         X

```

So we know we'll eventually do:

```python
pe[:, 0::2] = torch.sin(...)
pe[:, 1::2] = torch.cos(...)

```

The question is: **sin and cos of what?**

---
# Step 2: Every position needs a number

Our positions are simply:

```text
0
1
2
3

```

So:

```python
position = torch.arange(max_len)

```

gives:

```text
tensor([0, 1, 2, 3])

```

But we want it vertically:

```text
[
  [0],
  [1],
  [2],
  [3]
]

```

So:

```python
position = torch.arange(max_len).unsqueeze(1)

```

Now:

```text
position 0 → [0]
position 1 → [1]
position 2 → [2]
position 3 → [3]

```

---
# Step 3: Why do we need different frequencies?

Imagine we simply did:

```python
sin(position)

```

Then every even embedding dimension would be identical:

```text
             dim 0      dim 2      dim 4

position 0   sin(0)     sin(0)     sin(0)
position 1   sin(1)     sin(1)     sin(1)
position 2   sin(2)     sin(2)     sin(2)
position 3   sin(3)     sin(3)     sin(3)

```

That would be pointless because dimensions `0`, `2`, and `4` would contain the same information.

So each **pair of dimensions** gets a different speed/frequency:

```text
dimensions 0 and 1 → frequency A
dimensions 2 and 3 → frequency B
dimensions 4 and 5 → frequency C

```

Think of three clocks:

```text
Clock A: moves FAST
Clock B: moves MEDIUM
Clock C: moves SLOW

```

So instead of:

```text
sin(position)

```

we do conceptually:

```text
sin(position × frequency)

```

For example, pretend our frequencies were:

```text
[1.0, 0.1, 0.01]

```

Not the actual values yet — just for understanding.

---
# Step 4: Now look at the shapes

Positions:

```text
[
  [0],
  [1],
  [2],
  [3]
]

```

Frequencies:

```text
[1.0, 0.1, 0.01]

```

Now multiply them:

```text
                frequencies

                1.0    0.1    0.01
                 ↓      ↓       ↓

position 0  →   0.0    0.0     0.00
position 1  →   1.0    0.1     0.01
position 2  →   2.0    0.2     0.02
position 3  →   3.0    0.3     0.03

```

That's what **broadcasting** does.

In Python:

```python
angles = position * frequencies

```

You started with:

```text
position.shape
[4, 1]

```

and:

```text
frequencies.shape
[3]

```

and get:

```text
angles.shape
[4, 3]

```

because every position is multiplied by every frequency.

---
# Step 5: Put those angles into the final 6 dimensions

Now we have:

```text
angles =

[
  [0.0, 0.0, 0.00],
  [1.0, 0.1, 0.01],
  [2.0, 0.2, 0.02],
  [3.0, 0.3, 0.03]
]

```

Apply sine:

```python
torch.sin(angles)

```

That gives a `[4, 3]` matrix.

Where does it go?

```python
pe[:, 0::2]

```

Also has shape `[4, 3]` because it selects:

```text
dimensions 0, 2, 4

```

So:

```python
pe[:, 0::2] = torch.sin(angles)

```

puts:

```text
             dim 0       dim 2       dim 4

position 0   sin(0.0)    sin(0.0)    sin(0.00)
position 1   sin(1.0)    sin(0.1)    sin(0.01)
position 2   sin(2.0)    sin(0.2)    sin(0.02)
position 3   sin(3.0)    sin(0.3)    sin(0.03)

```

Then the **same angles** go through cosine into the odd columns:

```python
pe[:, 1::2] = torch.cos(angles)

```

Final result:

```text
             dim0       dim1       dim2       dim3       dim4       dim5

position 0   sin(a)      cos(a)      sin(b)      cos(b)      sin(c)      cos(c)
position 1   sin(a)      cos(a)      sin(b)      cos(b)      sin(c)      cos(c)
position 2   sin(a)      cos(a)      sin(b)      cos(b)      sin(c)      cos(c)
position 3   sin(a)      cos(a)      sin(b)      cos(b)      sin(c)      cos(c)

              └── pair ──┘
                                  └── pair ──┘
                                                       └── pair ──┘

```
## The whole mental model

Don't think about the scary formula yet. Think:

```text
1. Create positions
   [0, 1, 2, 3]

2. Create different frequencies
   [fast, medium, slow]

3. Combine every position with every frequency

             fast   medium   slow
   pos 0       x       x       x
   pos 1       x       x       x
   pos 2       x       x       x
   pos 3       x       x       x

4. sin → even dimensions
5. cos → odd dimensions

```

The `10000` formula is **just how the Transformer chooses **`[fast, medium, slow]`. Everything else is essentially tensor manipulation.