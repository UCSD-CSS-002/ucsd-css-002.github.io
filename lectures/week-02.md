# W2 problem

This week our ultimate goal is to take two books from [project gutenberg](https://www.gutenberg.org/): [Alice in Wonderland](https://www.gutenberg.org/ebooks/11), and [Frankenstein](https://www.gutenberg.org/ebooks/84), and figure out which words are disproportionately common in each book.  This kind of analysis is the first step to generating meaningful [word clouds](https://en.wikipedia.org/wiki/Tag_cloud) from text, and it underlies the family of [term frequency - inverse document frequency](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) methods commonly used for text search, user modeling, etc.

  

## Pseudo-algorithm

### Very high level

The gist of what we want to do is: 

- read in text of books
- count how often each word occurred in each book 
- for each word, figure out if it occurred more often in one book than the other.

This is very high level, and we need to specify the details of how we accomplish this.

### More detail

I. For each book:   
    1. read in book text   
    2. separate text into words   
    3. Loop through all the words.       
        a. if word is new: initialize its counter at 1.     
        b. if word is old: increment its counter by 1.     

II. For each word:   
    1. compare counters for that word from the two books.

This is getting better, but we are missing some key details of what exactly we want to do:  
- how do we read in files?  
- what are words?  how do we figure out if they are new or old?  
- how do we represent counters for each word?
- how do we compare counters?

### Files

How do we read in files?  We will cover that this week, looking at how to use the `open()` command to read in simple text files.

### Representing Counters

We will learn how to use the *dictionary* data structure to keep track of different counters for different words.

### Words

The computer will read in a large block of text as a single string.  How do we tell a computer what a "word" is?  How do we figure out which words are the same?  We will learn to work with *strings* to try to get words out, but we also have to define, conceptually, what we mean by a word!

Here is a sentence from Frankenstein: 

```
There—for with your leave, my sister, I will put some trust in preceding 
navigators—there snow and frost are banished; and, sailing over a calm sea, we may 
be wafted to a land surpassing in wonders and in beauty every region hitherto 
discovered on the habitable globe.
```

How do we decide what a word is? And which words are the same? 
The simplest thing to do is say that spaces divide strings into words and two words are the same if their strings are the same. 
So the above text ought to be divided into words accordingly: 

```
There—for
with
your
leave,
...
navigators—there
...
and
banished;
and, 
...
globe.
```

Note a few undesirable outcomes:  
- non alphabetic characters are included as part of a word (e.g., 'and' and 'and,' are different)    
- non space word breaks are disregarded (e.g., 'navigators-there' is one word)   
- words have varying capitalization (so 'There' and 'there' are different strings)  

So we might modify our procedure to say that we (1) convert all letters to lower case, (2) replace all non-letter characters with spaces, then (3) use spaces to split the string into words:  

converting to lower case and replacing non-letter characters yields: 

```
there for with your leave  my sister  I will put some trust in preceding 
navigators there snow and frost are banished  and  sailing over a calm sea  we may 
be wafted to a land surpassing in wonders and in beauty every region hitherto 
discovered on the habitable globe 
```

Which yields:  

```
yhere 
for 
with 
your 
leave
...
navigators
there
...
...
and
banished

and

...
globe
```

This looks much better.  We appear to have some words with no letters, because we had two spaces in a row, but we can deal with that.  

### Comparing counters

We said we will learn how to use *dictionaries* to store counters for different words.  But how do we compare counters between books?

For instance, Alice in wonderland and Frankenstein have the following counts for a few selected words.  

| Word | Alice in Wonderland | Frankenstein |
|-----|-----|-----|
| the | 1815 | 4366 |
| and | 911 | 3035 |
| to | 802 | 2173 |
| alice | 385 | 0 |
| my | 58 | 1773 | 
| pie | 2 | 0 | 

Note that if we just take the difference in counts, we will find that:  
- Frankenstein has more of just about every word (because frankenstein contains 77k words, while alice in wonderland has about 29k)    
- differences in counts are larger for more common words: (the occurs 2551 more times in frankenstein than alice in wonderland, while 'my' only occurs 1715 times more, even though 'my' should be more diagnostic, because frankenstein is written in first person while alice in wonderland is not!)  
- ratios of counts cannot be calculated because we often have counts of 0, and we cannot divide by 0.  

To address all of these problems, it is customary to calculate differences in counts by:
1. add 1 to the counter of each word being considered (this is called smoothing, so nothing has a frequency of exactly 0)   
2. divide the count by the total count, so we have a proportional frequency, rather than an absolute frequency.  
3. calculate the ratio of these frequencies.  
4. log10 transform the ratio of these frequencies to make the numbers more legible.  

Using this kind of procedure, we will find that words like 'elizabeth', and 'human', occur much more often in frankenstein, while words like 'alice', and 'hatter' occur much more often in alice in wonderland.  Furthmore, we find other disproportionalities, for words like ('dont' and 'im'), and ('during' and 'although'), which reflect different writing styles form different eras.

