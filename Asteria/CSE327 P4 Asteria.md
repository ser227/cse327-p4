```python
import sys
!{sys.executable} kbencoder.py -g --kb_path asteria_kb.txt
```

    Generating facts list...
    Forward chaining depth: 1
    New facts: 1020
    Forward chaining depth: 2
    New facts: 962
    Forward chaining depth: 3
    New facts: 224
    Forward chaining depth: 4
    New facts: 85
    Forward chaining depth: 5
    
    Facts list generated
    2291 total facts. Max depth = 4
    
    Number of queries: 200
    100 training queries generated
    100 testing queries generated



```python
import matplotlib.pyplot as plt
```


```python
!{sys.executable} kbencoder.py -g --kb_path asteria_kb.dl --new_vocab --vocab_from_kb --save_vocab
!head a.txt
```


```python
import sys
!{sys.executable} kbencoder.py --kb_path asteria_kb.txt --vocab_from_kb --save_vocab
```

    Creating vocabulary from last asteria_kb.txt knowledge base



```python
import sys
!{sys.executable} kbencoder.py --kb_path asteria_kb.txt --train_unification_model
```

    Using cuda device
    Uni model embed size: 50
     [████████████████████] 3500/3500 Anchors generated                                                 
     [████████████████████] 70000/70000 Triplets generated                                              
     [█████---------------] 18176/70000 Triplet encodings                                               

    IOPub data rate exceeded.
    The Jupyter server will temporarily stop sending output
    to the client in order to avoid crashing it.
    To change this limit, set the config variable
    `--ServerApp.iopub_data_rate_limit`.
    
    Current values:
    ServerApp.iopub_data_rate_limit=1000000.0 (bytes/sec)
    ServerApp.rate_limit_window=3.0 (secs)
    


     [████████████--------] 42383/70000 Triplet encodings                                               

    IOPub data rate exceeded.
    The Jupyter server will temporarily stop sending output
    to the client in order to avoid crashing it.
    To change this limit, set the config variable
    `--ServerApp.iopub_data_rate_limit`.
    
    Current values:
    ServerApp.iopub_data_rate_limit=1000000.0 (bytes/sec)
    ServerApp.rate_limit_window=3.0 (secs)
    


     [██████████████████--] 65940/70000 Triplet encodings                                               

    IOPub data rate exceeded.
    The Jupyter server will temporarily stop sending output
    to the client in order to avoid crashing it.
    To change this limit, set the config variable
    `--ServerApp.iopub_data_rate_limit`.
    
    Current values:
    ServerApp.iopub_data_rate_limit=1000000.0 (bytes/sec)
    ServerApp.rate_limit_window=3.0 (secs)
    


    Early stopping triggered-------------] 28.3% Epoch | [Epoch 142/500, Training Loss: 0.0725, Validation Loss: 0.3137]
    Saved model.
    Saved plot.
    unification model generated



```python
import glob
glob.glob('training_loss*.png')
```




    ['training_loss-10-100-5-50.png',
     'training_loss-50-96-2-50.png',
     'training_loss-45-106-2-50.png',
     'training_loss-10-100-4-50.png']




```python
img = plt.imread('training_loss-50-96-2-50.png')
plt. imshow(img)
```




    <matplotlib.image.AxesImage at 0x7d4976fed3d0>




    
![png](output_6_1.png)
    



```python
import sys
!{sys.executable} kbencoder.py --kb_path asteria_kb.txt -g
```

    Generating facts list...
    Forward chaining depth: 1
    New facts: 1020
    Forward chaining depth: 2
    New facts: 962
    Forward chaining depth: 3
    New facts: 224
    Forward chaining depth: 4
    New facts: 85
    Forward chaining depth: 5
    
    Facts list generated
    2291 total facts. Max depth = 4
    
    Number of queries: 200
    100 training queries generated
    100 testing queries generated



```python
import sys
!{sys.executable}  kbencoder.py --kb_path asteria_kb.txt -p
```

    running training queries...
    Negative facts will be generated
    1: [can_travel(H, S)]
    min depth: 8                                                                                        
    Nodes: 11811                                                                                        
    Answers: 1000                                                                                       
    
    2: [same_region_person(R1, L3)]
    Restart...                                                                                          
    Restart...                                                                                          
    Answers: 0                                                                                          
    
    3: [ancestor(C, L3)]
    min depth: 2                                                                                        
    Nodes: 38781                                                                                        
    Answers: 72                                                                                         
    
    4: [grandparent(G, B)]
    min depth: 3                                                                                        
    Nodes: 144                                                                                          
    Answers: 70                                                                                         
    
    5: [market_in(B, A), lives_in(talia_lunaris, A), trade_reach(B, kingdom_asteria)]
    min depth: 1                                                                                        
    Nodes: 7                                                                                            
    Answers: 6                                                                                          
    
    6: [influence(R1, Z)]
    min depth: 8                                                                                        
    min depth: 7                                                                                        
    Nodes: 18704                                                                                        
    Answers: 112                                                                                        
    
    7: [same_house(L, H2)]
    min depth: 3                                                                                        
    Nodes: 434                                                                                          
    Answers: 384                                                                                        
    
    8: [same_region_person(B, A), same_house(seris_solaris, A), lives_in(B, goldmarket)]
    min depth: 7                                                                                        
    Nodes: 8177                                                                                         
    Answers: 1000                                                                                       
    
    9: [same_house(A, brom_emberfall), lives_in(A, emberfall_hold)]
    min depth: 3                                                                                        
    Nodes: 61                                                                                           
    Answers: 11                                                                                         
    
    10: [can_travel(B, A), can_travel(darius_solaris, A), same_region_person(C, B), spouse(B, C)]
    min depth: 5                                                                                        
    Nodes: 4786                                                                                         
    Answers: 456                                                                                        
    
    11: [bonds_with(aelric_solaris, A), war_beast(A), dragon(A)]
    min depth: 1                                                                                        
    Nodes: 2                                                                                            
    Answers: 1                                                                                          
    
    12: [can_travel(B, D)]
    min depth: 4                                                                                        
    Nodes: 4386                                                                                         
    Answers: 400                                                                                        
    
    13: [descendant(R, L)]
    min depth: 5                                                                                        
    Nodes: 5096                                                                                         
    Answers: 148                                                                                        
    
    14: [connected_region(S, Z)]
    min depth: 4                                                                                        
    Nodes: 5270                                                                                         
    Answers: 702                                                                                        
    
    15: [fealty_chain(L, H1)]
    min depth: 2                                                                                        
    Nodes: 25291                                                                                        
    Answers: 33                                                                                         
    
    16: [child(L1, X)]
    min depth: 2                                                                                        
    Nodes: 74                                                                                           
    Answers: 72                                                                                         
    
    17: [resident_of(A, B), regional_subject(A, B), regional_subject(C, B), same_region_person(evan_riverwyn, C)]
    min depth: 4                                                                                        
    Nodes: 207                                                                                          
    Answers: 96                                                                                         
    
    18: [same_house(B, A), same_house(cassian_lunaris, A), serves(vessa_lunaris, B)]
    min depth: 3                                                                                        
    Nodes: 434                                                                                          
    Answers: 384                                                                                        
    
    19: [adult(L)]
    min depth: 2                                                                                        
    Nodes: 51                                                                                           
    Answers: 48                                                                                         
    
    20: [overlord(B, A), member_of_house(C, A), house_member(evan_riverwyn, B), alive(C)]
    min depth: 4                                                                                        
    Nodes: 286                                                                                          
    Answers: 10                                                                                         
    
    21: [same_region_person(selene_lunaris, A), child(B, A), regional_subject(B, C), regional_subject(selene_lunaris, C)]
    min depth: 7                                                                                        
    Nodes: 1543                                                                                         
    Answers: 28                                                                                         
    
    22: [same_house(L, R1)]
    min depth: 3                                                                                        
    Nodes: 434                                                                                          
    Answers: 384                                                                                        
    
    23: [same_house(B, B)]
    min depth: 3                                                                                        
    Nodes: 98                                                                                           
    Answers: 48                                                                                         
    
    24: [descendant(C, B)]
    min depth: 3                                                                                        
    Nodes: 15887                                                                                        
    Answers: 72                                                                                         
    
    25: [same_house(H2, L1)]
    min depth: 3                                                                                        
    Nodes: 434                                                                                          
    Answers: 384                                                                                        
    
    26: [fealty_chain(C, A)]
    min depth: 2                                                                                        
    Nodes: 38209                                                                                        
    Answers: 33                                                                                         
    
    27: [same_region_person(S, A)]
    min depth: 7                                                                                        
    Nodes: 8430                                                                                         
    Answers: 1000                                                                                       
    
    28: [mother(B, X)]
    min depth: 3                                                                                        
    Nodes: 110                                                                                          
    Answers: 36                                                                                         
    
    29: [member_of_order(seris_solaris, A), member_of_order(vessa_lunaris, A)]
    min depth: 1                                                                                        
    Nodes: 2                                                                                            
    Answers: 1                                                                                          
    
    30: [same_region_person(B, A), mother(A, maeve_solaris), same_house(A, A), same_region_person(orion_solaris, B), ancestor(A, B)]
    min depth: 9                                                                                        
    Nodes: 8948                                                                                         
    Answers: 1000                                                                                       
    
    31: [person(H1)]
    min depth: 2                                                                                        
    Nodes: 51                                                                                           
    Answers: 48                                                                                         
    
    32: [same_house(L3, L3)]
    min depth: 3                                                                                        
    Nodes: 98                                                                                           
    Answers: 48                                                                                         
    
    33: [child(D, X)]
    min depth: 2                                                                                        
    Nodes: 74                                                                                           
    Answers: 72                                                                                         
    
    34: [same_region_person(R2, B)]
    min depth: 7                                                                                        
    Nodes: 8929                                                                                         
    Answers: 1000                                                                                       
    
    35: [influence(A, B), influence(A, southern_realm), connected_region(B, B), resident_of(aelric_solaris, B), sworn_to(house_lunaris, A)]
    min depth: 5                                                                                        
    Nodes: 4560                                                                                         
    Answers: 192                                                                                        
    
    36: [adult(H1)]
    min depth: 2                                                                                        
    Nodes: 51                                                                                           
    Answers: 48                                                                                         
    
    37: [same_region_person(G, R2)]
    min depth: 9                                                                                        
    Nodes: 9861                                                                                         
    Answers: 1000                                                                                       
    
    38: [same_house(H2, L3)]
    min depth: 3                                                                                        
    Nodes: 434                                                                                          
    Answers: 384                                                                                        
    
    39: [grandparent(alina_riverwyn, A), same_region_person(A, maris_riverwyn)]
    min depth: 3                                                                                        
    Nodes: 80                                                                                           
    Answers: 6                                                                                          
    
    40: [can_travel(H, B)]
    min depth: 8                                                                                        
    min depth: 6                                                                                        
    Nodes: 9531                                                                                         
    Answers: 1000                                                                                       
    
    41: [same_region_person(C, Y)]
    min depth: 9                                                                                        
    Nodes: 7824                                                                                         
    Answers: 1000                                                                                       
    
    42: [house_member(A, house_emberfall), same_house(A, A)]
    min depth: 2                                                                                        
    Nodes: 10                                                                                           
    Answers: 8                                                                                          
    
    43: [can_travel(Y, R)]
    Restart...                                                                                          
    min depth: 8                                                                                        
    Nodes: 37599                                                                                        
    Answers: 1000                                                                                       
    
    44: [descendant(R1, R2)]
    Restart...                                                                                          
    min depth: 3                                                                                        
    Nodes: 843                                                                                          
    Answers: 146                                                                                        
    
    45: [fealty_chain(A, maris_riverwyn), serves(A, maris_riverwyn), child(A, maris_riverwyn)]
    Restart...                                                                                          
    Restart...                                                                                          
    Answers: 0                                                                                          
    
    46: [adult(D)]
    min depth: 2                                                                                        
    Nodes: 51                                                                                           
    Answers: 48                                                                                         
    
    47: [same_region_person(S, L3)]
    min depth: 7                                                                                        
    Nodes: 6545                                                                                         
    Answers: 712                                                                                        
    
    48: [same_region_person(C, A), ancestor(A, B), same_house(D, B), father(C, briar_thornvale), same_house(D, darian_stoneward)]
    min depth: 9                                                                                        
    Nodes: 9461                                                                                         
    Answers: 1000                                                                                       
    
    49: [can_travel(L1, G)]
    min depth: 4                                                                                        
    Nodes: 3734                                                                                         
    Answers: 452                                                                                        
    
    50: [female(A), same_house(B, A), same_region_person(B, C), child(C, cassian_lunaris)]
    min depth: 1                                                                                        
    Nodes: 26                                                                                           
    Answers: 25                                                                                         
    
    51: [eligible_council(H)]
    min depth: 2                                                                                        
    Nodes: 200                                                                                          
    Answers: 51                                                                                         
    
    52: [can_travel(R1, L)]
    min depth: 6                                                                                        
    Nodes: 13616                                                                                        
    Answers: 984                                                                                        
    
    53: [same_house(A, A), can_travel(A, B), can_travel(C, B), same_house(C, elara_solaris)]
    min depth: 3                                                                                        
    Nodes: 98                                                                                           
    Answers: 48                                                                                         
    
    54: [same_house(A, Z)]
    min depth: 3                                                                                        
    Nodes: 434                                                                                          
    Answers: 384                                                                                        
    
    55: [same_region_person(D, H2)]
    min depth: 9                                                                                        
    Nodes: 8772                                                                                         
    Answers: 1000                                                                                       
    
    56: [can_travel(R1, H)]
    min depth: 6                                                                                        
    Nodes: 12116                                                                                        
    Answers: 1000                                                                                       
    
    57: [can_travel(R2, L2)]
    Restart...                                                                                          
    min depth: 4                                                                                        
    Nodes: 3109                                                                                         
    Answers: 403                                                                                        
    
    58: [lives_in(A, emberfall_hold), spouse(A, kaela_emberfall)]
    min depth: 1                                                                                        
    Nodes: 7                                                                                            
    Answers: 6                                                                                          
    
    59: [can_travel(A, B), same_region_person(A, A), can_travel(lyra_thornvale, B)]
    min depth: 5                                                                                        
    Nodes: 3984                                                                                         
    Answers: 400                                                                                        
    
    60: [market_in(A, C), trade_reach(A, B), can_travel(tessa_stoneward, B), lives_in(D, C), same_house(D, kara_stoneward)]
    min depth: 1                                                                                        
    Nodes: 7                                                                                            
    Answers: 6                                                                                          
    
    61: [ancestor(C, R2)]
    Restart...                                                                                          
    min depth: 4                                                                                        
    Nodes: 100104                                                                                       
    Answers: 110                                                                                        
    
    62: [same_house(B, A), regional_subject(A, thornwood), ancestor(B, wren_thornvale)]
    min depth: 3                                                                                        
    Nodes: 434                                                                                          
    Answers: 384                                                                                        
    
    63: [resident_of(R2, H1)]
    min depth: 4                                                                                        
    Nodes: 687                                                                                          
    Answers: 96                                                                                         
    
    64: [descendant(S, R2)]
    min depth: 5                                                                                        
    Nodes: 12547                                                                                        
    Answers: 134                                                                                        
    
    65: [market_in(C, A), located_in(A, cinderlands), lives_in(B, A), child(B, myra_emberfall), trade_reach(C, kingdom_asteria)]
    min depth: 1                                                                                        
    Nodes: 7                                                                                            
    Answers: 6                                                                                          
    
    66: [same_region_person(L3, R2)]
    min depth: 9                                                                                        
    Nodes: 8898                                                                                         
    Answers: 1000                                                                                       
    
    67: [lives_in(liora_solaris, A), located_in(A, sunreach)]
    min depth: 1                                                                                        
    Nodes: 2                                                                                            
    Answers: 1                                                                                          
    
    68: [same_region_person(A, G)]
    min depth: 7                                                                                        
    Nodes: 9125                                                                                         
    Answers: 1000                                                                                       
    
    69: [lives_in(faye_thornvale, A), located_in(A, B), can_travel(talia_lunaris, B), lives_in(briar_thornvale, A)]
    min depth: 1                                                                                        
    Nodes: 2                                                                                            
    Answers: 1                                                                                          
    
    70: [eligible_council(L2)]
    min depth: 2                                                                                        
    Nodes: 200                                                                                          
    Answers: 51                                                                                         
    
    71: [noble(L)]
    min depth: 2                                                                                        
    Nodes: 50                                                                                           
    Answers: 48                                                                                         
    
    72: [same_house(A, galen_stoneward), can_travel(A, B), resident_of(cedric_thornvale, B), same_region_person(A, owen_stoneward)]
    min depth: 3                                                                                        
    Nodes: 57                                                                                           
    Answers: 7                                                                                          
    
    73: [resident_of(B, A), part_of(A, kingdom_asteria), regional_subject(elara_solaris, A), influence(house_solaris, A), ancestor(aelric_solaris, B)]
    min depth: 4                                                                                        
    Nodes: 170                                                                                          
    Answers: 96                                                                                         
    
    74: [same_house(R2, S)]
    min depth: 3                                                                                        
    Nodes: 434                                                                                          
    Answers: 384                                                                                        
    
    75: [descendant(R1, A)]
    min depth: 5                                                                                        
    Nodes: 33875                                                                                        
    Answers: 110                                                                                        
    
    76: [same_region_person(H, H1)]
    min depth: 7                                                                                        
    Nodes: 7741                                                                                         
    Answers: 1000                                                                                       
    
    77: [can_travel(L1, H)]
    min depth: 4                                                                                        
    Nodes: 3955                                                                                         
    Answers: 465                                                                                        
    
    78: [same_region_person(G, X)]
    min depth: 7                                                                                        
    Nodes: 8787                                                                                         
    Answers: 1000                                                                                       
    
    79: [same_region_person(B, B)]
    min depth: 7                                                                                        
    Nodes: 1131                                                                                         
    Answers: 116                                                                                        
    
    80: [same_house(H2, Y)]
    min depth: 3                                                                                        
    Nodes: 434                                                                                          
    Answers: 384                                                                                        
    
    81: [descendant(A, C), child(A, B), same_house(A, A), same_house(B, B), alive(C)]
    min depth: 5                                                                                        
    Nodes: 9979                                                                                         
    Answers: 140                                                                                        
    
    82: [fealty_chain(X, Z)]
    Restart...                                                                                          
    Restart...                                                                                          
    min depth: 2                                                                                        
    Nodes: 39286                                                                                        
    Answers: 33                                                                                         
    
    83: [can_travel(A, moonfen), same_region_person(A, cedric_thornvale)]
    Nodes: 100061                                                                                       
    Answers: 159                                                                                        
    
    84: [ancestor(aelric_solaris, A), same_house(A, seris_solaris)]
    Restart...                                                                                          
    Restart...                                                                                          
    Answers: 0                                                                                          
    
    85: [can_travel(G, S)]
    min depth: 6                                                                                        
    min depth: 4                                                                                        
    Nodes: 14489                                                                                        
    Answers: 920                                                                                        
    
    86: [can_travel(Y, G)]
    min depth: 8                                                                                        
    Nodes: 12722                                                                                        
    Answers: 1000                                                                                       
    
    87: [same_region_person(X, A)]
    min depth: 9                                                                                        
    Nodes: 9044                                                                                         
    Answers: 1000                                                                                       
    
    88: [regional_subject(H, L3)]
    min depth: 7                                                                                        
    min depth: 5                                                                                        
    Nodes: 6673                                                                                         
    Answers: 91                                                                                         
    
    89: [same_region_person(darian_stoneward, A), same_region_person(B, A), father(A, petra_stoneward), same_region_person(bram_stoneward, B), descendant(B, rowan_thornvale)]
    min depth: 9                                                                                        
    Nodes: 192                                                                                          
    Answers: 32                                                                                         
    
    90: [can_travel(A, moonfen), can_travel(A, stonecoast)]
    min depth: 6                                                                                        
    min depth: 4                                                                                        
    Nodes: 10343                                                                                        
    Answers: 222                                                                                        
    
    91: [descendant(B, A), same_region_person(faye_thornvale, A), same_region_person(tessa_stoneward, B)]
    Restart...                                                                                          
    min depth: 5                                                                                        
    Nodes: 10877                                                                                        
    Answers: 142                                                                                        
    
    92: [age_group(A, B), regional_subject(A, kingdom_asteria), age_group(elara_solaris, B), grandparent(A, faye_thornvale)]
    min depth: 1                                                                                        
    Nodes: 49                                                                                           
    Answers: 48                                                                                         
    
    93: [same_region_person(B, A), same_region_person(leon_lunaris, A), same_house(B, C), parent(nyra_lunaris, C)]
    min depth: 7                                                                                        
    Nodes: 8773                                                                                         
    Answers: 1000                                                                                       
    
    94: [person(A), same_house(A, B), fealty_chain(B, selene_lunaris), grandparent(nyra_lunaris, B)]
    min depth: 2                                                                                        
    Nodes: 51                                                                                           
    Answers: 48                                                                                         
    
    95: [same_house(nella_riverwyn, A), regional_subject(A, B), resident_of(C, B), same_house(C, nella_riverwyn)]
    min depth: 3                                                                                        
    Nodes: 11                                                                                           
    Answers: 8                                                                                          
    
    96: [bonds_with(toren_emberfall, A), war_beast(A)]
    min depth: 1                                                                                        
    Nodes: 2                                                                                            
    Answers: 1                                                                                          
    
    97: [same_region_person(darius_solaris, A), same_region_person(A, seris_solaris), serves(A, aelric_solaris)]
    min depth: 7                                                                                        
    Nodes: 1393                                                                                         
    Answers: 12                                                                                         
    
    98: [serves(evan_riverwyn, A), mother(A, B), parent(A, mira_riverwyn), same_region_person(B, A), same_house(B, A)]
    min depth: 1                                                                                        
    Nodes: 3                                                                                            
    Answers: 2                                                                                          
    
    99: [child(S, R2)]
    min depth: 2                                                                                        
    Nodes: 74                                                                                           
    Answers: 72                                                                                         
    
    100: [child(A, R)]
    min depth: 2                                                                                        
    Nodes: 74                                                                                           
    Answers: 72                                                                                         
    
    Before removing duplicates: 1261234 examples
    After removing duplicates: 21414 examples
    Replicating pos examples 1 times
    pos len=15393, neg len=15393



```python
!{sys.executable} nnreasoner.py --embed_type unification
```

    States from vocab: vocab
    Num predicates: 50
    Max arity: 2
    Num constants: 96
    Num variables: 19
    
    Training unification
    Embed size: 50
    Reading examples from mr_train_examples.csv
    Parsing examples...
    Generating embeddings...
    Embedding input size: 280
     [███████████████] 30785/30785 Prepping examples                                                    
    Loaded data...
    Using cuda device
    0	0.67926 (-)
    50	0.50066 (-0.0037)
    100	0.43769 (-0.0024)
    150	0.41102 (-0.0018)
    200	0.39160 (-0.0015)
    250	0.37750 (-0.0012)
    300	0.37103 (-0.0010)
    350	0.36603 (-0.0009)
    400	0.36117 (-0.0008)
    450	0.35527 (-0.0007)
    500	0.35007 (-0.0006)
    550	0.34769 (-0.0006)
    600	0.34376 (-0.0005)
    650	0.34273 (-0.0005)
    700	0.33989 (-0.0005)
    750	0.34030 (-0.0004)
    800	0.33888 (-0.0004)
    850	0.33500 (-0.0004)
    900	0.33577 (-0.0004)
    950	0.33341 (-0.0003)
    1000	0.33155 (-0.0003)
    1050	0.33419 (-0.0003)
    1100	0.33104 (-0.0003)
    1150	0.32886 (-0.0003)
    1200	0.33119 (-0.0003)
    1250	0.32713 (-0.0002)
    1300	0.32635 (-0.0002)
    1350	0.32761 (-0.0002)
    1400	0.32642 (-0.0002)
    1450	0.32357 (-0.0002)
    1500	0.32299 (-0.0002)
    1550	0.32617 (-0.0002)
    1600	0.32270 (-0.0002)
    1650	0.32294 (-0.0002)
    1700	0.32072 (-0.0002)
    1750	0.32131 (-0.0002)
    1800	0.32137 (-0.0002)
    1850	0.31978 (-0.0002)
    1900	0.31867 (-0.0001)
    1950	0.31697 (-0.0001)
    2000	0.31876 (-0.0001)
    2050	0.31652 (-0.0001)
    2100	0.31679 (-0.0001)
    2150	0.31584 (-0.0001)
    2200	0.33159 (-0.0001)
    2250	0.32584 (-0.0001)
    2300	0.32540 (-0.0001)
    2349	0.32408 (-0.0001)
    Saved training loss figure.



```python
img = plt.imread('guided_loss-unification50-96-2-50.png')
plt. imshow(img)
```




    <matplotlib.image.AxesImage at 0x7d49a3f67f10>




    
![png](output_10_1.png)
    



```python
!{sys.executable}  evaluate.py --kb asteria_kb.txt -s
!{sys.executable} evaluate.py --kb asteria_kb.txt -u
```

    using cuda device
    STANDARD
    
    Query 1: [same_house(H1, H2)]
    4 :: 3 - 00:00:00 (3818 nps)                                                                        
    
    Query 2: [descendant(G, R1)]
    4 :: 3 - 00:00:00 (2572 nps)                                                                        
    
    Query 3: [same_region_person(L1, D)]
    8 :: 7 - 00:00:00 (5581 nps)                                                                        
    
    Query 4: [mother(A, caelan_solaris), same_house(A, B), same_region_person(seris_solaris, A), descendant(A, B), same_region_person(A, B)]
    Query failed!!!                                                                                     
    199 :: 0 - 00:00:00 (13194 nps)
    
    Query 5: [can_travel(G, H2)]
    5 :: 4 - 00:00:00 (6622 nps)                                                                        
    
    Query 6: [descendant(C, L)]
    4 :: 3 - 00:00:00 (4266 nps)                                                                        
    
    Query 7: [member_of_house(A, B), same_house(galen_stoneward, A), house_member(galen_stoneward, B), parent(A, C), same_region_person(tessa_stoneward, C)]
    111 :: 14 - 00:00:00 (7187 nps)                                                                     
    
    Query 8: [regional_subject(Y, G)]
    6 :: 5 - 00:00:00 (7034 nps)                                                                        
    
    Query 9: [same_region_person(R, D)]
    8 :: 7 - 00:00:00 (11095 nps)                                                                       
    
    Query 10: [same_house(R, R)]
    4 :: 3 - 00:00:00 (4418 nps)                                                                        
    
    Query 11: [alive(A), same_region_person(B, A), same_region_person(B, C), same_region_person(B, isolde_thornvale), same_house(C, galen_stoneward)]
    Query failed!!!                                                                                     
    140609 :: 0 - 00:00:02 (68680 nps)
    
    Query 12: [same_house(C, R)]
    4 :: 3 - 00:00:00 (3982 nps)                                                                        
    
    Query 13: [same_region_person(B, L)]
    8 :: 7 - 00:00:00 (6144 nps)                                                                        
    
    Query 14: [trade_reach(R2, C)]
    4 :: 3 - 00:00:00 (13558 nps)                                                                       
    
    Query 15: [regional_subject(H, L2)]
    6 :: 5 - 00:00:00 (2375 nps)                                                                        
    
    Query 16: [same_region_person(G, R)]
    8 :: 7 - 00:00:00 (10891 nps)                                                                       
    
    Query 17: [fealty_chain(L2, X)]
    3 :: 2 - 00:00:00 (6288 nps)                                                                        
    
    Query 18: [regional_subject(A, riverlands), same_house(B, A), same_region_person(C, B), same_house(C, kellan_riverwyn)]
    Query failed!!!                                                                                     
    44850 :: 0 - 00:00:00 (107189 nps)
    
    Query 19: [same_region_person(B, A), adult(A), lives_in(A, greenhollow), same_region_person(B, rowan_thornvale)]
    Query failed!!!                                                                                     
    18212 :: 0 - 00:00:00 (61116 nps)
    
    Query 20: [resident_of(X, B)]
    4 :: 3 - 00:00:00 (5374 nps)                                                                        
    
    Query 21: [same_region_person(L, L)]
    8 :: 7 - 00:00:00 (6124 nps)                                                                        
    
    Query 22: [can_travel(C, L1)]
    5 :: 4 - 00:00:00 (6336 nps)                                                                        
    
    Query 23: [house_member(B, A), rules_house(B, A), spouse(B, myra_emberfall), same_house(brom_emberfall, B), fealty_chain(garrick_emberfall, B)]
    30 :: 9 - 00:00:00 (9349 nps)                                                                       
    
    Query 24: [resident_of(S, Y)]
    4 :: 3 - 00:00:00 (814 nps)                                                                         
    
    Query 25: [same_house(L1, L1)]
    4 :: 3 - 00:00:00 (4757 nps)                                                                        
    
    Query 26: [regional_subject(L, X)]
    6 :: 5 - 00:00:00 (7754 nps)                                                                        
    
    Query 27: [descendant(ash_thornvale, A), same_region_person(A, tessa_stoneward)]
    23 :: 12 - 00:00:00 (8924 nps)                                                                      
    
    Query 28: [regional_subject(C, A), regional_subject(B, A), male(B), same_region_person(C, C)]
    Query failed!!!                                                                                     
    79610 :: 0 - 00:00:00 (95246 nps)
    
    Query 29: [child(H2, R1)]
    3 :: 2 - 00:00:00 (3220 nps)                                                                        
    
    Query 30: [road(moonfen, A), connected_region(stonecoast, A), resident_of(evan_riverwyn, A)]
    7 :: 6 - 00:00:00 (8021 nps)                                                                        
    
    Query 31: [same_house(L, R2)]
    4 :: 3 - 00:00:00 (6350 nps)                                                                        
    
    Query 32: [same_house(B, H2)]
    4 :: 3 - 00:00:00 (6563 nps)                                                                        
    
    Query 33: [same_region_person(D, H1)]
    8 :: 7 - 00:00:00 (6114 nps)                                                                        
    
    Query 34: [member_of_order(A, grove_circle), serves(cedric_thornvale, A)]
    3 :: 2 - 00:00:00 (8410 nps)                                                                        
    
    Query 35: [can_travel(R1, Z)]
    Query failed!!!                                                                                     
    295 :: 0 - 00:00:00 (27108 nps)
    
    Query 36: [member_of_order(A, grove_circle), member_of_house(A, house_thornvale)]
    3 :: 2 - 00:00:00 (7222 nps)                                                                        
    
    Query 37: [child(L2, L)]
    3 :: 2 - 00:00:00 (3390 nps)                                                                        
    
    Query 38: [child(maris_riverwyn, A), same_house(nella_riverwyn, A), age_group(A, B), age_group(C, B), alive(C)]
    9 :: 8 - 00:00:00 (3565 nps)                                                                        
    
    Query 39: [regional_subject(H2, X)]
    6 :: 5 - 00:00:00 (9264 nps)                                                                        
    
    Query 40: [sworn_to(house_riverwyn, A), house_member(maeve_solaris, A), house_member(darius_solaris, A)]
    6 :: 5 - 00:00:00 (7438 nps)                                                                        
    
    Query 41: [age_group(A, elder), father(A, B), parent(B, leon_lunaris)]
    19 :: 5 - 00:00:00 (2295 nps)                                                                       
    
    Query 42: [can_travel(A, riverlands), can_travel(A, B), connected_region(B, B)]
    38 :: 12 - 00:00:00 (177 nps)                                                                       
    
    Query 43: [same_region_person(L, H2)]
    8 :: 7 - 00:00:00 (11100 nps)                                                                       
    
    Query 44: [same_house(A, B), same_region_person(A, rowan_thornvale), same_region_person(B, ash_thornvale)]
    Query failed!!!                                                                                     
    8818 :: 0 - 00:00:00 (81226 nps)
    
    Query 45: [same_house(L, L3)]
    4 :: 3 - 00:00:00 (7915 nps)                                                                        
    
    Query 46: [seat_of(B, A), lives_in(D, A), house_member(C, B), same_region_person(C, orion_solaris), same_region_person(D, D)]
    Query failed!!!                                                                                     
    6127 :: 0 - 00:00:00 (88860 nps)
    
    Query 47: [same_region_person(A, maeve_solaris), same_region_person(A, darius_solaris)]
    15 :: 14 - 00:00:00 (25457 nps)                                                                     
    
    Query 48: [descendant(nash_stoneward, A), child(A, B), regional_subject(A, southern_realm), can_travel(B, cinderlands)]
    Query failed!!!                                                                                     
    556 :: 0 - 00:00:00 (7079 nps)
    
    Query 49: [same_region_person(R2, H1)]
    8 :: 7 - 00:00:00 (10770 nps)                                                                       
    
    Query 50: [mother(R, C)]
    5 :: 3 - 00:00:00 (3768 nps)                                                                        
    
    Query 51: [same_house(A, elara_solaris), father(B, A), same_region_person(liora_solaris, B)]
    20 :: 13 - 00:00:00 (6262 nps)                                                                      
    
    Query 52: [regional_subject(kaela_emberfall, A), regional_subject(garrick_emberfall, A)]
    11 :: 10 - 00:00:00 (8443 nps)                                                                      
    
    Query 53: [ancestor(toren_emberfall, A), mother(myra_emberfall, A), regional_subject(A, kingdom_asteria), can_travel(A, stonecoast)]
    Query failed!!!                                                                                     
    78 :: 0 - 00:00:00 (7978 nps)
    
    Query 54: [fealty_chain(D, L2)]
    3 :: 2 - 00:00:00 (5832 nps)                                                                        
    
    Query 55: [can_travel(G, H)]
    5 :: 4 - 00:00:00 (17630 nps)                                                                       
    
    Query 56: [age_group(B, A), age_group(C, A), ancestor(B, caelan_solaris), child(briar_thornvale, C)]
    350 :: 8 - 00:00:00 (20219 nps)                                                                     
    
    Query 57: [same_house(B, A), mother(A, C), same_house(iris_lunaris, B), child(talia_lunaris, C), lives_in(C, moonwatch_castle)]
    394 :: 12 - 00:00:00 (16451 nps)                                                                    
    
    Query 58: [influence(A, cinderlands), member_of_house(B, A), resident_of(B, cinderlands)]
    106 :: 9 - 00:00:00 (9655 nps)                                                                      
    
    Query 59: [resident_of(R, R2)]
    4 :: 3 - 00:00:00 (5196 nps)                                                                        
    
    Query 60: [same_region_person(B, A), mother(A, owen_stoneward), same_region_person(A, briar_thornvale), same_region_person(lyra_thornvale, B)]
    Query failed!!!                                                                                     
    16284 :: 0 - 00:00:00 (70186 nps)
    
    Query 61: [member_of_order(B, A), descendant(B, corin_riverwyn)]
    191 :: 4 - 00:00:00 (6822 nps)                                                                      
    
    Query 62: [lives_in(A, B), age_group(A, adult), market_in(granite_guild, B)]
    70 :: 3 - 00:00:00 (4673 nps)                                                                       
    
    Query 63: [same_region_person(A, H1)]
    8 :: 7 - 00:00:00 (42377 nps)                                                                       
    
    Query 64: [lives_in(A, sunspire_keep), ancestor(B, A), same_region_person(C, B), can_travel(C, riverlands)]
    Query failed!!!                                                                                     
    13539 :: 0 - 00:00:00 (40311 nps)
    
    Query 65: [same_house(A, B), same_region_person(A, A), can_travel(B, stonecoast), can_travel(B, riverlands), can_travel(B, cinderlands)]
    Query failed!!!                                                                                     
    18290 :: 0 - 00:00:00 (63995 nps)
    
    Query 66: [can_travel(Y, X)]
    5 :: 4 - 00:00:00 (5952 nps)                                                                        
    
    Query 67: [child(H2, H1)]
    3 :: 2 - 00:00:00 (2574 nps)                                                                        
    
    Query 68: [same_region_person(D, H)]
    8 :: 7 - 00:00:00 (10253 nps)                                                                       
    
    Query 69: [father(B, L1)]
    4 :: 3 - 00:00:00 (4001 nps)                                                                        
    
    Query 70: [same_house(A, B), lives_in(A, rivermeet_hall), person(B)]
    223 :: 6 - 00:00:00 (21487 nps)                                                                     
    
    Query 71: [mother(L2, Z)]
    5 :: 3 - 00:00:00 (5709 nps)                                                                        
    
    Query 72: [parent(B, A), can_travel(A, stonecoast), can_travel(B, moonfen)]
    2493 :: 13 - 00:00:00 (8420 nps)                                                                    
    
    Query 73: [same_region_person(L, Z)]
    8 :: 7 - 00:00:00 (9540 nps)                                                                        
    
    Query 74: [age_group(A, elder), same_region_person(A, B), same_house(darius_solaris, B)]
    12 :: 11 - 00:00:00 (7664 nps)                                                                      
    
    Query 75: [same_region_person(C, R1)]
    Query failed!!!                                                                                     
    436 :: 0 - 00:00:00 (50599 nps)
    
    Query 76: [same_region_person(H, L1)]
    8 :: 7 - 00:00:00 (5343 nps)                                                                        
    
    Query 77: [same_house(A, B), magic_user(A), same_house(nyra_lunaris, A), resident_of(B, moonfen)]
    206 :: 11 - 00:00:00 (51037 nps)                                                                    
    
    Query 78: [can_travel(nash_stoneward, A), can_travel(lyra_thornvale, A), can_travel(B, A), same_house(B, B), same_region_person(B, B)]
    Query failed!!!                                                                                     
    2906 :: 0 - 00:00:00 (40049 nps)
    
    Query 79: [descendant(A, cassian_lunaris), same_house(B, A), same_region_person(B, talia_lunaris), same_region_person(A, nyra_lunaris)]
    Query failed!!!                                                                                     
    739 :: 0 - 00:00:00 (36395 nps)
    
    Query 80: [lives_in(A, thornvale_manor), same_region_person(briar_thornvale, A), same_region_person(faye_thornvale, A)]
    16 :: 15 - 00:00:00 (11334 nps)                                                                     
    
    Query 81: [same_region_person(L2, D)]
    8 :: 7 - 00:00:00 (11429 nps)                                                                       
    
    Query 82: [trade_reach(L2, R2)]
    4 :: 3 - 00:00:00 (13558 nps)                                                                       
    
    Query 83: [trade_reach(A, B), trade_reach(A, C), within_region(northern_realm, B), regional_subject(leon_lunaris, B), influence(crown_of_asteria, C)]
    Query failed!!!                                                                                     
    669 :: 0 - 00:00:00 (18310 nps)
    
    Query 84: [can_travel(D, L3)]
    5 :: 4 - 00:00:00 (16339 nps)                                                                       
    
    Query 85: [controls_region(R, C)]
    4 :: 3 - 00:00:00 (13492 nps)                                                                       
    
    Query 86: [same_region_person(darius_solaris, A), parent(A, maeve_solaris)]
    9 :: 8 - 00:00:00 (8042 nps)                                                                        
    
    Query 87: [descendant(G, R)]
    4 :: 3 - 00:00:00 (3959 nps)                                                                        
    
    Query 88: [same_region_person(R2, H2)]
    8 :: 7 - 00:00:00 (52627 nps)                                                                       
    
    Query 89: [eligible_council(Z)]
    7 :: 5 - 00:00:00 (5124 nps)                                                                        
    
    Query 90: [connected_region(X, C)]
    3 :: 2 - 00:00:00 (12413 nps)                                                                       
    
    Query 91: [same_house(L3, H2)]
    4 :: 3 - 00:00:00 (5230 nps)                                                                        
    
    Query 92: [market_in(B, A), lives_in(kara_stoneward, A), trade_reach(B, C), trade_reach(B, stonecoast), can_travel(ash_thornvale, C)]
    25 :: 14 - 00:00:00 (9546 nps)                                                                      
    
    Query 93: [person(A), mother(B, A), descendant(A, B), same_house(maris_riverwyn, B)]
    273 :: 11 - 00:00:00 (3737 nps)                                                                     
    
    Query 94: [trade_reach(L, R2)]
    4 :: 3 - 00:00:00 (18676 nps)                                                                       
    
    Query 95: [can_travel(A, sunreach), same_house(A, maeve_solaris)]
    8 :: 7 - 00:00:00 (7470 nps)                                                                        
    
    Query 96: [same_house(H1, R1)]
    4 :: 3 - 00:00:00 (8730 nps)                                                                        
    
    Query 97: [mother(R1, D)]
    5 :: 3 - 00:00:00 (4736 nps)                                                                        
    
    Query 98: [person(H2)]
    3 :: 2 - 00:00:00 (10798 nps)                                                                       
    
    Query 99: [grandparent(B, C)]
    4 :: 3 - 00:00:00 (2569 nps)                                                                        
    
    Query 100: [influence(house_riverwyn, A), within_region(A, kingdom_asteria), can_travel(tessa_stoneward, A), can_travel(kellan_riverwyn, A)]
    Query failed!!!                                                                                     
    547 :: 0 - 00:00:00 (41678 nps)
    
    std: 3577.23 nodes/query
    18 queries failed
    Time to run all queries: 5.68577294799999
    using cuda device
    UNITY: ming
    	Embedding Model: rKB_model.pth
    	Scoring Model: uni_mr_model.pt
    
    	Control: mingoal
    	Rule Eval: Max Rule
    Learned Embedding: Network using cuda
    Query 1: [same_house(H1, H2)]
    4 :: 3 - 00:00:00 (21 nps)                                                                          
    
    Query 2: [descendant(G, R1)]
    4 :: 3 - 00:00:00 (14 nps)                                                                          
    
    Query 3: [same_region_person(L1, D)]
    8 :: 7 - 00:00:00 (23 nps)                                                                          
    
    Query 4: [mother(A, caelan_solaris), same_house(A, B), same_region_person(seris_solaris, A), descendant(A, B), same_region_person(A, B)]
    Query failed!!!                                                                                     
    612 :: 0 - 00:00:08 (70 nps)
    
    Query 5: [can_travel(G, H2)]
    5 :: 4 - 00:00:00 (24 nps)                                                                          
    
    Query 6: [descendant(C, L)]
    4 :: 3 - 00:00:00 (74 nps)                                                                          
    
    Query 7: [member_of_house(A, B), same_house(galen_stoneward, A), house_member(galen_stoneward, B), parent(A, C), same_region_person(tessa_stoneward, C)]
    15 :: 14 - 00:00:00 (16 nps)                                                                        
    
    Query 8: [regional_subject(Y, G)]
    8 :: 7 - 00:00:00 (56 nps)                                                                          
    
    Query 9: [same_region_person(R, D)]
    11 :: 7 - 00:00:00 (31 nps)                                                                         
    
    Query 10: [same_house(R, R)]
    3 :: 2 - 00:00:00 (18 nps)                                                                          
    
    Query 11: [alive(A), same_region_person(B, A), same_region_person(B, C), same_region_person(B, isolde_thornvale), same_house(C, galen_stoneward)]
    Query failed!!!                                                                                     
    36179 :: 0 - 00:00:13 (2686 nps)
    
    Query 12: [same_house(C, R)]
    4 :: 3 - 00:00:00 (96 nps)                                                                          
    
    Query 13: [same_region_person(B, L)]
    8 :: 7 - 00:00:00 (28 nps)                                                                          
    
    Query 14: [trade_reach(R2, C)]
    4 :: 3 - 00:00:00 (237 nps)                                                                         
    
    Query 15: [regional_subject(H, L2)]
    8 :: 7 - 00:00:00 (49 nps)                                                                          
    
    Query 16: [same_region_person(G, R)]
    11 :: 7 - 00:00:00 (255 nps)                                                                        
    
    Query 17: [fealty_chain(L2, X)]
    3 :: 2 - 00:00:00 (110 nps)                                                                         
    
    Query 18: [regional_subject(A, riverlands), same_house(B, A), same_region_person(C, B), same_house(C, kellan_riverwyn)]
    149424 :: 15 - 00:00:25 (5762 nps)                                                                  
    
    Query 19: [same_region_person(B, A), adult(A), lives_in(A, greenhollow), same_region_person(B, rowan_thornvale)]
    437 :: 15 - 00:00:02 (176 nps)                                                                      
    
    Query 20: [resident_of(X, B)]
    4 :: 3 - 00:00:00 (32 nps)                                                                          
    
    Query 21: [same_region_person(L, L)]
    5 :: 4 - 00:00:00 (1423 nps)                                                                        
    
    Query 22: [can_travel(C, L1)]
    5 :: 4 - 00:00:00 (301 nps)                                                                         
    
    Query 23: [house_member(B, A), rules_house(B, A), spouse(B, myra_emberfall), same_house(brom_emberfall, B), fealty_chain(garrick_emberfall, B)]
    53292 :: 9 - 00:00:11 (4725 nps)                                                                    
    
    Query 24: [resident_of(S, Y)]
    4 :: 3 - 00:00:00 (11 nps)                                                                          
    
    Query 25: [same_house(L1, L1)]
    3 :: 2 - 00:00:00 (72 nps)                                                                          
    
    Query 26: [regional_subject(L, X)]
    8 :: 7 - 00:00:00 (148 nps)                                                                         
    
    Query 27: [descendant(ash_thornvale, A), same_region_person(A, tessa_stoneward)]
    31 :: 12 - 00:00:00 (143 nps)                                                                       
    
    Query 28: [regional_subject(C, A), regional_subject(B, A), male(B), same_region_person(C, C)]
    28 :: 15 - 00:00:00 (28 nps)                                                                        
    
    Query 29: [child(H2, R1)]
    3 :: 2 - 00:00:00 (47 nps)                                                                          
    
    Query 30: [road(moonfen, A), connected_region(stonecoast, A), resident_of(evan_riverwyn, A)]
    7 :: 6 - 00:00:00 (219 nps)                                                                         
    
    Query 31: [same_house(L, R2)]
    4 :: 3 - 00:00:00 (93 nps)                                                                          
    
    Query 32: [same_house(B, H2)]
    4 :: 3 - 00:00:00 (26 nps)                                                                          
    
    Query 33: [same_region_person(D, H1)]
    8 :: 7 - 00:00:00 (14 nps)                                                                          
    
    Query 34: [member_of_order(A, grove_circle), serves(cedric_thornvale, A)]
    3 :: 2 - 00:00:00 (761 nps)                                                                         
    
    Query 35: [can_travel(R1, Z)]
    46 :: 6 - 00:00:00 (175 nps)                                                                        
    
    Query 36: [member_of_order(A, grove_circle), member_of_house(A, house_thornvale)]
    3 :: 2 - 00:00:00 (400 nps)                                                                         
    
    Query 37: [child(L2, L)]
    3 :: 2 - 00:00:00 (14 nps)                                                                          
    
    Query 38: [child(maris_riverwyn, A), same_house(nella_riverwyn, A), age_group(A, B), age_group(C, B), alive(C)]
    9 :: 8 - 00:00:00 (11 nps)                                                                          
    
    Query 39: [regional_subject(H2, X)]
    6 :: 5 - 00:00:00 (14 nps)                                                                          
    
    Query 40: [sworn_to(house_riverwyn, A), house_member(maeve_solaris, A), house_member(darius_solaris, A)]
    6 :: 5 - 00:00:00 (543 nps)                                                                         
    
    Query 41: [age_group(A, elder), father(A, B), parent(B, leon_lunaris)]
    8 :: 5 - 00:00:00 (65 nps)                                                                          
    
    Query 42: [can_travel(A, riverlands), can_travel(A, B), connected_region(B, B)]
    91430 :: 14 - 00:00:29 (3111 nps)                                                                   
    
    Query 43: [same_region_person(L, H2)]
    8 :: 7 - 00:00:00 (26 nps)                                                                          
    
    Query 44: [same_house(A, B), same_region_person(A, rowan_thornvale), same_region_person(B, ash_thornvale)]
    Query failed!!!                                                                                     
    22832 :: 0 - 00:00:11 (2012 nps)
    
    Query 45: [same_house(L, L3)]
    4 :: 3 - 00:00:00 (110 nps)                                                                         
    
    Query 46: [seat_of(B, A), lives_in(D, A), house_member(C, B), same_region_person(C, orion_solaris), same_region_person(D, D)]
    8844 :: 14 - 00:00:03 (2375 nps)                                                                    
    
    Query 47: [same_region_person(A, maeve_solaris), same_region_person(A, darius_solaris)]
    31 :: 14 - 00:00:00 (53 nps)                                                                        
    
    Query 48: [descendant(nash_stoneward, A), child(A, B), regional_subject(A, southern_realm), can_travel(B, cinderlands)]
    Query failed!!!                                                                                     
    26471 :: 0 - 00:00:23 (1117 nps)
    
    Query 49: [same_region_person(R2, H1)]
    Query failed!!!                                                                                     
    1856 :: 0 - 00:00:00 (1922 nps)
    
    Query 50: [mother(R, C)]
    4 :: 3 - 00:00:00 (55 nps)                                                                          
    
    Query 51: [same_house(A, elara_solaris), father(B, A), same_region_person(liora_solaris, B)]
    14 :: 13 - 00:00:00 (14 nps)                                                                        
    
    Query 52: [regional_subject(kaela_emberfall, A), regional_subject(garrick_emberfall, A)]
    8 :: 7 - 00:00:00 (84 nps)                                                                          
    
    Query 53: [ancestor(toren_emberfall, A), mother(myra_emberfall, A), regional_subject(A, kingdom_asteria), can_travel(A, stonecoast)]
    Query failed!!!                                                                                     
    128085 :: 0 - 00:01:06 (1940 nps)
    
    Query 54: [fealty_chain(D, L2)]
    3 :: 2 - 00:00:00 (44 nps)                                                                          
    
    Query 55: [can_travel(G, H)]
    5 :: 4 - 00:00:00 (65 nps)                                                                          
    
    Query 56: [age_group(B, A), age_group(C, A), ancestor(B, caelan_solaris), child(briar_thornvale, C)]
    18 :: 8 - 00:00:01 (15 nps)                                                                         
    
    Query 57: [same_house(B, A), mother(A, C), same_house(iris_lunaris, B), child(talia_lunaris, C), lives_in(C, moonwatch_castle)]
    203 :: 12 - 00:00:00 (275 nps)                                                                      
    
    Query 58: [influence(A, cinderlands), member_of_house(B, A), resident_of(B, cinderlands)]
    295636 :: 9 - 00:00:41 (7116 nps)                                                                   
    
    Query 59: [resident_of(R, R2)]
    4 :: 3 - 00:00:00 (79 nps)                                                                          
    
    Query 60: [same_region_person(B, A), mother(A, owen_stoneward), same_region_person(A, briar_thornvale), same_region_person(lyra_thornvale, B)]
    Query failed!!!                                                                                     
    992 :: 0 - 00:00:03 (256 nps)
    
    Query 61: [member_of_order(B, A), descendant(B, corin_riverwyn)]
    365533 :: 4 - 00:00:43 (8490 nps)                                                                   
    
    Query 62: [lives_in(A, B), age_group(A, adult), market_in(granite_guild, B)]
    39 :: 3 - 00:00:00 (195 nps)                                                                        
    
    Query 63: [same_region_person(A, H1)]
    8 :: 7 - 00:00:01 (7 nps)                                                                           
    
    Query 64: [lives_in(A, sunspire_keep), ancestor(B, A), same_region_person(C, B), can_travel(C, riverlands)]
    Query failed!!!                                                                                     
    1000000 :: 0 - 00:01:07 (14785 nps)
    
    Query 65: [same_house(A, B), same_region_person(A, A), can_travel(B, stonecoast), can_travel(B, riverlands), can_travel(B, cinderlands)]
    Query failed!!!                                                                                     
    1000000 :: 0 - 00:05:17 (3148 nps)
    
    Query 66: [can_travel(Y, X)]
    5 :: 4 - 00:00:00 (102 nps)                                                                         
    
    Query 67: [child(H2, H1)]
    3 :: 2 - 00:00:00 (47 nps)                                                                          
    
    Query 68: [same_region_person(D, H)]
    31 :: 7 - 00:00:00 (64 nps)                                                                         
    
    Query 69: [father(B, L1)]
    4 :: 3 - 00:00:00 (27 nps)                                                                          
    
    Query 70: [same_house(A, B), lives_in(A, rivermeet_hall), person(B)]
    64 :: 6 - 00:00:00 (252 nps)                                                                        
    
    Query 71: [mother(L2, Z)]
    4 :: 3 - 00:00:00 (54 nps)                                                                          
    
    Query 72: [parent(B, A), can_travel(A, stonecoast), can_travel(B, moonfen)]
    328787 :: 13 - 00:00:32 (10244 nps)                                                                 
    
    Query 73: [same_region_person(L, Z)]
    8 :: 7 - 00:00:00 (25 nps)                                                                          
    
    Query 74: [age_group(A, elder), same_region_person(A, B), same_house(darius_solaris, B)]
    46 :: 11 - 00:00:00 (81 nps)                                                                        
    
    Query 75: [same_region_person(C, R1)]
    Query failed!!!                                                                                     
    1856 :: 0 - 00:00:01 (1491 nps)
    
    Query 76: [same_region_person(H, L1)]
    31 :: 7 - 00:00:00 (711 nps)                                                                        
    
    Query 77: [same_house(A, B), magic_user(A), same_house(nyra_lunaris, A), resident_of(B, moonfen)]
    17 :: 11 - 00:00:00 (145 nps)                                                                       
    
    Query 78: [can_travel(nash_stoneward, A), can_travel(lyra_thornvale, A), can_travel(B, A), same_house(B, B), same_region_person(B, B)]
    Query failed!!!                                                                                     
    25513 :: 0 - 00:00:42 (601 nps)
    
    Query 79: [descendant(A, cassian_lunaris), same_house(B, A), same_region_person(B, talia_lunaris), same_region_person(A, nyra_lunaris)]
    Query failed!!!                                                                                     
    1728 :: 0 - 00:00:50 (34 nps)
    
    Query 80: [lives_in(A, thornvale_manor), same_region_person(briar_thornvale, A), same_region_person(faye_thornvale, A)]
    77 :: 14 - 00:00:00 (117 nps)                                                                       
    
    Query 81: [same_region_person(L2, D)]
    8 :: 7 - 00:00:00 (9 nps)                                                                           
    
    Query 82: [trade_reach(L2, R2)]
    4 :: 3 - 00:00:00 (116 nps)                                                                         
    
    Query 83: [trade_reach(A, B), trade_reach(A, C), within_region(northern_realm, B), regional_subject(leon_lunaris, B), influence(crown_of_asteria, C)]
    Query failed!!!                                                                                     
    39258 :: 0 - 00:01:23 (470 nps)
    
    Query 84: [can_travel(D, L3)]
    5 :: 4 - 00:00:00 (309 nps)                                                                         
    
    Query 85: [controls_region(R, C)]
    4 :: 3 - 00:00:00 (31 nps)                                                                          
    
    Query 86: [same_region_person(darius_solaris, A), parent(A, maeve_solaris)]
    9 :: 8 - 00:00:00 (269 nps)                                                                         
    
    Query 87: [descendant(G, R)]
    4 :: 3 - 00:00:00 (71 nps)                                                                          
    
    Query 88: [same_region_person(R2, H2)]
    Query failed!!!                                                                                     
    1082 :: 0 - 00:00:00 (1198 nps)
    
    Query 89: [eligible_council(Z)]
    3 :: 2 - 00:00:00 (790 nps)                                                                         
    
    Query 90: [connected_region(X, C)]
    3 :: 2 - 00:00:00 (268 nps)                                                                         
    
    Query 91: [same_house(L3, H2)]
    4 :: 3 - 00:00:00 (17 nps)                                                                          
    
    Query 92: [market_in(B, A), lives_in(kara_stoneward, A), trade_reach(B, C), trade_reach(B, stonecoast), can_travel(ash_thornvale, C)]
    20 :: 13 - 00:00:00 (26 nps)                                                                        
    
    Query 93: [person(A), mother(B, A), descendant(A, B), same_house(maris_riverwyn, B)]
    304 :: 9 - 00:00:01 (286 nps)                                                                       
    
    Query 94: [trade_reach(L, R2)]
    4 :: 3 - 00:00:00 (21 nps)                                                                          
    
    Query 95: [can_travel(A, sunreach), same_house(A, maeve_solaris)]
    361 :: 11 - 00:00:00 (512 nps)                                                                      
    
    Query 96: [same_house(H1, R1)]
    4 :: 3 - 00:00:00 (28 nps)                                                                          
    
    Query 97: [mother(R1, D)]
    4 :: 3 - 00:00:00 (34 nps)                                                                          
    
    Query 98: [person(H2)]
    3 :: 2 - 00:00:00 (154 nps)                                                                         
    
    Query 99: [grandparent(B, C)]
    5 :: 3 - 00:00:00 (19 nps)                                                                          
    
    Query 100: [influence(house_riverwyn, A), within_region(A, kingdom_asteria), can_travel(tessa_stoneward, A), can_travel(kellan_riverwyn, A)]
    Query failed!!!                                                                                     
    330390 :: 0 - 00:00:54 (6085 nps)
    
    ming: 39119.23 nodes/query
    15 queries failed
    Time to run all queries: 955.5795072099999



```python
!ls *.csv
```

    ming-45-106-2-50.csv  mr_train_examples.csv  std-50-96-2-50.csv
    ming-50-96-2-50.csv   std-45-106-2-50.csv    triplets.csv



```python
!head std-50-96-2-50.csv 
!head ming-50-96-2-50.csv
```

    query,std reasoner,std nodes explored,std min depth,success,time
    1,std,4,3,True,0.0010476360000009066
    2,std,4,3,True,0.0015551939999998154
    3,std,8,7,True,0.001433352999999471
    4,std,199,0,False,0.015081998999999513
    5,std,5,4,True,0.0007550349999991823
    6,std,4,3,True,0.0009375009999992301
    7,std,111,14,True,0.015442679999999598
    8,std,6,5,True,0.0008529970000008547
    9,std,8,7,True,0.0007210249999989315
    query,ming reasoner,ming nodes explored,ming min depth,success,time
    1,ming,4,3,True,0.18988686699999935
    2,ming,4,3,True,0.26940185199999966
    3,ming,8,7,True,0.33363928200000004
    4,ming,612,0,False,8.681280444
    5,ming,5,4,True,0.2058537660000006
    6,ming,4,3,True,0.05339731300000139
    7,ming,15,14,True,0.8901303180000006
    8,ming,8,7,True,0.14180567900000085
    9,ming,11,7,True,0.35226056800000194



```python

```
