explain select count(*) from title t, movie_companies mc, movie_keyword mk where t.id=mc.movie_id and t.id=mk.movie_id and t.kind_id = 7 and t.production_year < 2013 and mc.company_type_id = 1;