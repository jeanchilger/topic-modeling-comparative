# Initial model analisys 

The first analisys was applied on 18 different models. Each model was a produced by permuting three hyperparameters, namely `kappa`, `minimum_probability` and `normalize` comprising the sets of values `{0.4, 0.9, 1.5}`, `{0.1, 0.02, 0.005}` and `{True, False}`, respectively. Following are all these models listed by name, a number was assigned to each one to ease their mention.

1. `kappa=0.4, minimum_probability=0.1, normalize=False`;
2. `kappa=0.4, minimum_probability=0.1, normalize=True`;
3. `kappa=0.4, minimum_probability=0.02, normalize=False`;
4. `kappa=0.4, minimum_probability=0.02, normalize=True`;
5. `kappa=0.4, minimum_probability=0.005, normalize=False`;
6. `kappa=0.4, minimum_probability=0.005, normalize=True`;
7. `kappa=0.9, minimum_probability=0.1, normalize=False`;
8. `kappa=0.9, minimum_probability=0.1, normalize=True`;
9. `kappa=0.9, minimum_probability=0.02, normalize=False`;
10. `kappa=0.9, minimum_probability=0.02, normalize=True`;
11. `kappa=0.9, minimum_probability=0.005, normalize=False`;
12. `kappa=0.9, minimum_probability=0.005, normalize=True`;
13. `kappa=1.5, minimum_probability=0.1, normalize=False`;
14. `kappa=1.5, minimum_probability=0.1, normalize=True`;
15. `kappa=1.5, minimum_probability=0.02, normalize=False`;
16. `kappa=1.5, minimum_probability=0.02, normalize=True`;
17. `kappa=1.5, minimum_probability=0.005, normalize=False`;
18. `kappa=1.5, minimum_probability=0.005, normalize=True`;

### Equal models
All models with same `kappa` value will not change at all. The weights for words in a topic changes if `normalize` changes. Between models with diferent `kappa`s the topic words changes slightly. The topic distribution also remains very similar through all models.

<!-- # Discussion on individual parameters

## `kappa`
## `minimum_probability`
## `normalize` -->

# Analisys on different corpora

Based on the similarity observed in the previously stated models, new models were generated, this time varying not only hyperparameters but also the corpus used in the training stage. Thus 16 new models were created. The different corpora used are as follows:

1. `only_phraser_nohtml`: Gensim's `Phraser` was used to generate bigrams, no other library was used for phrase detection. Itens between HTML tags (and the tags itself) were ignored;

2. `only_phraser_3_gram_nohtml`: Gensim's `Phraser` was used to generate trigrams, no other library was used for phrase detection. Itens between HTML tags (and the tags itself) were ignored;

3. `only_phraser_4_gram_nohtml`: Gensim's `Phraser` was used to generate fourgrams, no other library was used for phrase detection. Itens between HTML tags (and the tags itself) were ignored;

4. `both_merges_nohtml`: Spacy pipeline was used to detect entities and noun chunks. Itens between HTML tags (and the tags itself) were ignored.

The produced models was assembled by a permutation of the hyperparameters `kappa` and `minimum_probability` with sets values `{0.4, 0.9}`, `{0.1, 0.5}`, respectively, combined with each of the three corpora. Following is a list of the models:

19. `kappa=0.4, minimum_probability=0.1, data_file=both_merges_nohtml`;
20. `kappa=0.4, minimum_probability=0.1, data_file=only_phraser_3_gram_nohtml`;
21. `kappa=0.4, minimum_probability=0.1, data_file=only_phraser_nohtml`;
22. `kappa=0.4, minimum_probability=0.5, data_file=both_merges_nohtml`;
23. `kappa=0.4, minimum_probability=0.5, data_file=only_phraser_3_gram_nohtml`;
24. `kappa=0.4, minimum_probability=0.5, data_file=only_phraser_nohtml`;
25. `kappa=0.9, minimum_probability=0.1, data_file=both_merges_nohtml`;
26. `kappa=0.9, minimum_probability=0.1, data_file=only_phraser_3_gram_nohtml`;
27. `kappa=0.9, minimum_probability=0.1, data_file=only_phraser_nohtml`;
28. `kappa=0.9, minimum_probability=0.5, data_file=both_merges_nohtml`;
29. `kappa=0.9, minimum_probability=0.5, data_file=only_phraser_3_gram_nohtml`;
30. `kappa=0.9, minimum_probability=0.5, data_file=only_phraser_nohtml`;

## Conclusions
- Again `minimum_probability` haven't impact on topic distributions;

# Evaluation

## Coherence Score
Four models was selected and the coherece score was obtained for each one. The result is given in the table below.

| Model Name                                                               | Score               |
|--------------------------------------------------------------------------|---------------------|
| kappa=0.4, minimum_probability=0.1, data_file=only_phraser_nohtml        | -1.1551872156234722 |
| kappa=0.4, minimum_probability=0.1, data_file=only_phraser_3_gram_nohtml | -1.2333059023745476 |
| kappa=0.4, minimum_probability=0.1, data_file=only_phraser_4_gram_nohtml | -1.3600669505721164 |
| kappa=0.4, minimum_probability=0.1, data_file=both_merges_nohtml         | -1.2068782531591495 |

From these results, the first model on the table was chosen for the final topic distribution generation.

<!-- 
2
19 0.066*"like" + 0.060*"think" + 0.056*"know" + 0.050*"come" + 0.047*"want" + 0.047*"thing" + 0.045*"community" + 0.042*"get" + 0.038*"year" + 0.038*"right"

3
18 0.167*"pandemic" + 0.127*"time" + 0.077*"know" + 0.072*"think" + 0.072*"like" + 0.060*"come" + 0.047*"family" + 0.047*"get" + 0.047*"thing" + 0.046*"want"

4
8 0.167*"time" + 0.108*"know" + 0.103*"day" + 0.089*"like" + 0.087*"think" + 0.081*"come" + 0.061*"get" + 0.055*"thing" + 0.054*"want" + 0.048*"lot"

both:
like, think, know, come, want, thing, get, 

2
community, year, right

3
pandemic, family, 

4
day, lot

3+4
time,  -->