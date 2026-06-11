#include "ffmap.h"

typedef struct {
    index_t R[3];
    index_t ofrom;
    index_t oto;
    index_t _invalid;
} hash_formfac_t;

static inline hash_formfac_t hash_formfac_from_tu_formfactor( tu_formfactor_t t,
        const index_t* nk ) {
    hash_formfac_t h = {0};
    for (int d=0; d<3; ++d)
        h.R[d] = t.R[d] > nk[d]/2 ? t.R[d]-nk[d] : t.R[d];
    h.ofrom = t.ofrom;
    h.oto = t.oto;
    return h;
}

#define vhash_val_t tu_formfactor_idx_t
#define vhash_val_empty tu_formfactor_idx_invalid
#define vhash_key_t hash_formfac_t
#define vhash_key_empty (hash_formfac_t){._invalid=1}
#define vhash_key_equal(A,B) (!memcmp(&(A),&(B),sizeof(hash_formfac_t)))

#include "vhashtable.h"
VHASH_IMPLEMENT( hash_formfac_ )

struct tu_formfactor_map_t {
    // first member to reinterpret the struct
    hash_formfac_table_t h;
    diverge_model_t* m;
};

tu_formfactor_map_t* tu_formfactor_map_init( diverge_model_t* m ) {
    hash_formfac_table_t* h = hash_formfac_table_init(m->n_tu_ff, 32);
    h = realloc(h, sizeof(tu_formfactor_map_t));
    tu_formfactor_map_t* tmap = (tu_formfactor_map_t*)h;
    tmap->m = m;
    for (index_t t=0; t<m->n_tu_ff; ++t)
        hash_formfac_table_insert(h, hash_formfac_from_tu_formfactor(m->tu_ff[t], m->nk),
                (tu_formfactor_idx_t){.t=t, .d=m->tu_ff[t].d});
    return tmap;
}

tu_formfactor_idx_t tu_formfactor_map_get( tu_formfactor_map_t* tmap,
        index_t Rx, index_t Ry, index_t Rz, index_t ofrom, index_t oto ) {
    return hash_formfac_table_find( &tmap->h, (hash_formfac_t){ .R={Rx,Ry,Rz}, .ofrom=ofrom, .oto=oto } );
}

tu_formfactor_idx_t tu_formfactor_map_get_tu( tu_formfactor_map_t* tmap,
        tu_formfactor_t t ) {
    return hash_formfac_table_find( &tmap->h, hash_formfac_from_tu_formfactor(t, tmap->m->nk) );
}

void tu_formfactor_map_free( tu_formfactor_map_t* tmap ) {
    hash_formfac_table_free( &tmap->h );
}

