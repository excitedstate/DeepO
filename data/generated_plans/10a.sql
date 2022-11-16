Aggregate  (cost=344940.28..344940.29 rows=1 width=64) (actual time=1201.453..1205.861 rows=1 loops=1)
  ->  Gather  (cost=5285.08..344940.27 rows=2 width=32) (actual time=198.954..1205.765 rows=41 loops=1)
        Workers Planned: 2
        Workers Launched: 2
        ->  Nested Loop  (cost=4285.08..343940.07 rows=1 width=32) (actual time=253.189..1182.868 rows=14 loops=3)
              ->  Nested Loop  (cost=4284.93..343939.50 rows=1 width=36) (actual time=252.979..1182.612 rows=14 loops=3)
                    ->  Nested Loop  (cost=4284.50..343935.27 rows=1 width=24) (actual time=252.382..1179.973 rows=15 loops=3)
                          ->  Nested Loop  (cost=4284.34..343922.90 rows=245 width=28) (actual time=250.210..1179.823 rows=24 loops=3)
                                Join Filter: (t.id = ci.movie_id)
                                ->  Nested Loop  (cost=4283.90..87652.37 rows=8880 width=28) (actual time=96.464..716.305 rows=1943 loops=3)
                                      ->  Parallel Hash Join  (cost=4283.47..65225.24 rows=16048 width=8) (actual time=96.309..577.909 rows=6981 loops=3)
                                            Hash Cond: (mc.company_id = cn.id)
                                            ->  Parallel Seq Scan on movie_companies mc  (cost=0.00..55518.57 rows=2065957 width=12) (actual time=0.339..361.103 rows=1652765 loops=3)
                                            ->  Parallel Hash  (cost=4262.78..4262.78 rows=1655 width=4) (actual time=92.867..92.868 rows=810 loops=3)
                                                  Buckets: 4096  Batches: 1  Memory Usage: 160kB
                                                  ->  Parallel Bitmap Heap Scan on company_name cn  (cost=34.22..4262.78 rows=1655 width=4) (actual time=17.112..92.068 rows=810 loops=3)
                                                        Recheck Cond: ((country_code)::text = '[ru]'::text)
                                                        Heap Blocks: exact=588
                                                        ->  Bitmap Index Scan on company_name_idx_ccode  (cost=0.00..33.52 rows=2813 width=0) (actual time=0.931..0.931 rows=2430 loops=1)
                                                              Index Cond: ((country_code)::text = '[ru]'::text)
                                      ->  Index Scan using title_pkey on title t  (cost=0.43..1.40 rows=1 width=20) (actual time=0.019..0.019 rows=0 loops=20944)
                                            Index Cond: (id = mc.movie_id)
                                            Filter: (production_year > 2005)
                                            Rows Removed by Filter: 0
                                ->  Index Scan using cast_info_idx_mid on cast_info ci  (cost=0.44..28.85 rows=1 width=12) (actual time=0.228..0.238 rows=0 loops=5830)
                                      Index Cond: (movie_id = mc.movie_id)
                                      Filter: ((note ~~ '%(voice)%'::text) AND (note ~~ '%(uncredited)%'::text))
                                      Rows Removed by Filter: 19
                          ->  Memoize  (cost=0.16..0.58 rows=1 width=4) (actual time=0.005..0.005 rows=1 loops=72)
                                Cache Key: ci.role_id
                                Cache Mode: logical
                                Hits: 30  Misses: 2  Evictions: 0  Overflows: 0  Memory Usage: 1kB
                                Worker 0:  Hits: 1  Misses: 2  Evictions: 0  Overflows: 0  Memory Usage: 1kB
                                Worker 1:  Hits: 34  Misses: 3  Evictions: 0  Overflows: 0  Memory Usage: 1kB
                                ->  Index Scan using role_type_pkey on role_type rt  (cost=0.15..0.57 rows=1 width=4) (actual time=0.036..0.036 rows=0 loops=7)
                                      Index Cond: (id = ci.role_id)
                                      Filter: ((role)::text = 'actor'::text)
                                      Rows Removed by Filter: 1
                    ->  Index Scan using char_name_pkey on char_name chn  (cost=0.43..4.23 rows=1 width=20) (actual time=0.170..0.170 rows=1 loops=46)
                          Index Cond: (id = ci.person_role_id)
              ->  Index Only Scan using company_type_pkey on company_type ct  (cost=0.15..0.57 rows=1 width=4) (actual time=0.014..0.014 rows=1 loops=41)
                    Index Cond: (id = mc.company_type_id)
                    Heap Fetches: 41
Planning Time: 27.229 ms
JIT:
  Functions: 143
  Options: Inlining false, Optimization false, Expressions true, Deforming true
  Timing: Generation 6.106 ms, Inlining 0.000 ms, Optimization 2.643 ms, Emission 48.243 ms, Total 56.992 ms
Execution Time: 1232.492 ms