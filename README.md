![training_plot_10000.png](training_plot_10000.png)

Ideas for fixing this 

1. Choose an easier problem lol

2. get rid of pooling. i think that might be rlly bad lol.

3. attention?

4. simpler state representation -- eg instead of feeding in the
   whole grid just feed in a little patch surrounding the
   character  -- oh that's kind of nice actually  and then i dont
   need to buy a gpu lol

5. let the agent remember its path.
for instance, the input could include the last couple places
where the guy has been or something

6. "ciriculum learning" -- start with easy mazes, work up to
   harder ones.
   this could look like having mazes which are mostly empty to
   start with :)


7. reward exploration for reals -- keep track of recently visited
   states, reward visiting new ones

---

oh dear. 
I think I made my reward hackable.
going back and forth gets infinite reward. sigh.
i hope we never do that irl

---

oh apparently a pretty principled way to do reward shaping is 
by adding $\gamma*\phi(S_{t+1}) - \phi(S_t)$.
Like this does not change $\argmax_\theta \E_{\tau\sim \pi_\theta} \sum_{t=0}^{T} \gamma^{t}r_t$ 
if the start and ending states are fixed.


---

```bash
git clone https://github.com/awestover/FDT; cd FDT; pip install -r requirements.txt
```


----

plan: 
1. try ciriculum learning
2. why is it so slow?
3. give up on this project :'(



```python
import cProfile
cProfile.run('main()', 'training_stats')

import pstats
p = pstats.Stats('training_stats')
p.sort_stats('cumulative').print_stats(30)
```
wow this code was useful!

