explain select count(*) from title t, movie_companies mc, cast_info ci where t.id=mc.movie_id and t.id=ci.movie_id and t.production_year > 1958;