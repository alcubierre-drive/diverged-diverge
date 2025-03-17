/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#define LINALG_NO_LAPACKE
#include "../misc/linalg.h"
LINALG_IMPLEMENT

// this must follow the LINALG include to pre-define the types
#include "diverge_interface.h"
#include "phonons.h"

#include "../misc/qnan.h"
#include "../diverge_model_internals.h"
#include "../misc/mergesort.h"
#include "../misc/btree.h"
#include <assert.h>

void diverge_model_internals_tu( diverge_model_t* model, double max_dist ) {
    if (!model->internals->has_common_internals) {
        mpi_err_printf("model needs common internals before tu internals!\n");
        diverge_model_internals_common(model);
    }
    strcpy(model->internals->backend, "tu");
    tu_data_t* info = calloc(1, sizeof*info);
    info->nk = model->nk[0] * model->nk[1] * model->nk[2];
    info->nkf = model->nkf[0] * model->nkf[1] * model->nkf[2];
    info->nktot = model->nk[0] * model->nk[1] * model->nk[2]*
                 model->nkf[0] * model->nkf[1] * model->nkf[2];
    info->symm = calloc(1, sizeof*info->symm);
    info->use_reduced_loop = true;
    info->use_cuda_mpi_loop = false;
    info->tu_selfenergy_flow = false;
    info->mpi_fft_chunksize = -1; // auto
    #if defined(USE_MPI) && defined(USE_MPI_FFTW)
    info->which_fft_simple = info->which_fft_greens = info->which_fft_red = tu_fft_algorithm_mpi_new;
    #else
    info->which_fft_simple = info->which_fft_greens = info->which_fft_red = tu_fft_algorithm_greedy;
    #endif
    info->propagator_extra_cpu_timings = false;
    model->internals->tu_data = (void*)info;
    model->internals->tu_data_destructor = tu_data_destructor;

    #ifdef DIVERGE_TU_GF_COARSIFY_FFT
    info->GF_fft_coarsify = true;
    #else
    info->GF_fft_coarsify = false;
    #endif

    info->selfenergy_circular_cutoff = false;

    init_ff_struct(max_dist, model);
    generate_reduced_loop_maps(model);
    tu_init_symmetries(model);
    info->offload_cpu = 0.0;
    info->percent_mem_GPU_blocked = 0.9;
}

static void tu_data_phonons_destructor( void* v ) {
    tu_data_phonons_free((tu_data_phonons_t*)v);
}
static void model_phonons_destructor( void* v ) {
    diverge_model_t* m = v;
    m->internals->tu_phonon_model_dont_free = 0;
    diverge_model_free(m);
}

diverge_model_t* diverge_model_internals_tu_phonons( diverge_model_t* model,
        double max_dist, rs_hopping_t* hop, index_t n_hop, rs_vertex_t* vert,
        index_t n_vert ) {
    diverge_model_internals_tu( model, max_dist );
    strcpy( model->internals->backend, "tu_phonons" );
    mpi_vrb_printf( "setting up phonon internals\n" );

    diverge_model_t* m = diverge_model_init();
    memcpy( m->nk, model->nk, sizeof(m->nk) );
    memcpy( m->nkf, model->nkf, sizeof(m->nkf) );
    for (int d=model->internals->dim; d<3; ++d) m->nk[d] = m->nkf[d] = 0;
    memcpy( m->lattice, model->lattice, sizeof(m->lattice) );
    m->hop = calloc(n_hop, sizeof*m->hop); memcpy( m->hop, hop, n_hop*sizeof*m->hop );
    m->n_hop = n_hop;
    m->vert = calloc(n_vert, sizeof*m->vert); memcpy( m->vert, vert, n_vert*sizeof*m->vert );
    m->n_vert = n_vert;
    index_t max_orb = 0;
    for (index_t i=0; i<n_hop; ++i) {
        max_orb = MAX(max_orb, m->hop[i].o1);
        max_orb = MAX(max_orb, m->hop[i].o2);
    }
    ph_vertex_t* pvert = (ph_vertex_t*)m->vert;
    for (index_t i=0; i<n_vert; ++i) {
        max_orb = MAX(max_orb, pvert[i].O1);
    }
    m->n_orb = max_orb + 1;
    m->n_spin = m->SU2 = 1;
    strncpy(m->name, model->name, sizeof(m->name));
    strncat(m->name, "_phonons", sizeof(m->name));
    m->name[sizeof m->name - 1] = 0;
    memcpy(&m->n_ibz_path, &model->n_ibz_path, sizeof m->n_ibz_path);
    memcpy(m->ibz_path, model->ibz_path, sizeof m->ibz_path);
    if (model->n_sym > 0) mpi_log_printf( "symmetries not added to phonon model\n" );
    if (model->n_orb > 0) mpi_log_printf( "nontrivial basis positions not implemented in phonon model\n" );
    diverge_model_internals_common(m);

    m->internals->tu_phonon_model_dont_free = 1;
    tu_data_phonons_t* p = tu_data_phonons_init( m, model, model->internals->tu_phonons_buffered );
    model->internals->tu_data_phonons = p;
    model->internals->tu_data_phonons_destructor = &tu_data_phonons_destructor;
    model->internals->model_phonons = m;
    model->internals->model_phonons_destructor = &model_phonons_destructor;
    return m;
}

void tu_data_destructor(void* v) {
    tu_data_t* tu_data = v;
    tu_symmstruct_delete( tu_data->symm );
    free(tu_data->bond_sizes);
    free(tu_data->bond_offsets);
    free(tu_data->ff_idx_to_unique_ff_elem);
    free(tu_data->ob_to_orbff);
    free(tu_data->mi_to_ofrom);
    free(tu_data->mi_to_ffidx);
    free(tu_data->mi_to_tuffidx);
    free(tu_data->mi_to_oto);
    free(tu_data->mi_to_R);

    free(tu_data->existing_ff);
    free(tu_data->existing_ff_fine);
    free(tu_data->unique_ff_combi);
    free(tu_data->unique_mi_l);
    free(tu_data->unique_mi_r);
    free(tu_data->lmap_l);
    free(tu_data->lmap_r);
    free(tu_data->lmap_off);
    free(tu_data->lmap_len);

    free(tu_data);
}


static index_t map_to_fine(index_t val, index_t nc) {
    if (labs(nc - val) < (nc / 2) + 1) val = val - nc;
    return val;
}

static inline void fill_existing_ff( diverge_model_t* model ) {
    tu_data_t* tu_data = (tu_data_t*)model->internals->tu_data;
    tu_formfactor_t* tu_ff = model->tu_ff;
    for (index_t i=0; i<model->n_tu_ff; ++i) {
        claV3i_t ff = {{tu_ff[i].R[0], tu_ff[i].R[1], tu_ff[i].R[2]}};
        index_t idx = 0;
        for (; idx<tu_data->existing_ff_sz; ++idx) {
            claV3i_t dd = claV3i_sub(tu_data->existing_ff[idx], ff);
            if (claV3i_dot(dd,dd) == 0)
                break;
        }
        if (idx == tu_data->existing_ff_sz) {
            VECTOR_PUSH_BACK( tu_data->existing_ff, ff );
        }
        tu_ff[i].ffidx = idx;
    }
    VECTOR_SHRINK_TO_FIT( tu_data->existing_ff );

    VECTOR_RESIZE( tu_data->existing_ff_fine, tu_data->existing_ff_sz );
    for (index_t i=0; i<tu_data->existing_ff_sz; ++i)
        for (int d=0; d<3; ++d)
            tu_data->existing_ff_fine[i].v[d] = map_to_fine(tu_data->existing_ff[i].v[d], model->nk[d]);
}

static inline int ffcomp( tu_formfactor_t x, tu_formfactor_t y ) {
    if (x.ofrom != y.ofrom)     return x.ofrom - y.ofrom;
    if (fabs(x.d-y.d) > 1e-7)   return x.d < y.d ? -1 : x.d > y.d ? 1 : 0;
    if (x.R[0] != y.R[0])       return x.R[0] - y.R[0];
    if (x.R[1] != y.R[1])       return x.R[1] - y.R[1];
    if (x.R[2] != y.R[2])       return x.R[2] - y.R[2];
    return labs(x.ofrom - x.oto) - labs(y.ofrom - y.oto);
}

#undef MERGESORT_COMP
#define MERGESORT_COMP(x,y) ffcomp(x,y)
MERGESORT_IMPL( mergesort_formfactors, tu_formfactor_t )

static inline void sort_ff_struct( tu_formfactor_t* tu_ff, index_t n_tu_ff ) {
    mergesort_formfactors( tu_ff, n_tu_ff, diverge_omp_num_threads(), 0 );
}

void init_bond_descriptor(diverge_model_t* model) {
    index_t n_tu_ff = model->n_tu_ff;
    tu_formfactor_t* tu_ff = model->tu_ff;
    tu_data_t* tu_data = (tu_data_t*)model->internals->tu_data;
    tu_data->bond_sizes = (index_t*)calloc(model->n_orb, sizeof(index_t));
    tu_data->bond_offsets = (index_t*)calloc(model->n_orb, sizeof(index_t));
    for(index_t i = 0; i < n_tu_ff; ++i)
        tu_data->bond_sizes[tu_ff[i].ofrom] += 1;

    for(index_t i = 1; i < model->n_orb; ++i)
        tu_data->bond_offsets[i] = tu_data->bond_offsets[i-1]+tu_data->bond_sizes[i-1];

    for (index_t i=0; i<model->n_orb; ++i) // find max bond size
        tu_data->n_bonds = MAX(tu_data->n_bonds, tu_data->bond_sizes[i]);
    tu_data->ob_to_orbff = (index_t*)calloc(model->n_orb * tu_data->n_bonds, sizeof(index_t));
    tu_data->n_orbff = n_tu_ff;
    tu_data->mi_to_ofrom = (index_t*)calloc(n_tu_ff,sizeof(index_t));
    tu_data->mi_to_oto = (index_t*)calloc(n_tu_ff,sizeof(index_t));
    tu_data->mi_to_R = (index_t*)calloc(3*n_tu_ff,sizeof(index_t));
    tu_data->mi_to_ffidx = (index_t*)calloc(n_tu_ff,sizeof(index_t));
    tu_data->mi_to_tuffidx = (index_t*)calloc(n_tu_ff,sizeof(index_t));
    index_t idx = 0;
    for(index_t b = 0; b < tu_data->n_bonds; ++b)
    for(index_t o = 0; o < model->n_orb; ++o){
        if(b < tu_data->bond_sizes[o]) {
            tu_data->ob_to_orbff[b + tu_data->n_bonds * o] = idx;
            tu_data->mi_to_ofrom[idx] = tu_ff[b+tu_data->bond_offsets[o]].ofrom;
            tu_data->mi_to_oto[idx]   = tu_ff[b+tu_data->bond_offsets[o]].oto;
            tu_data->mi_to_ffidx[idx] = tu_ff[b+tu_data->bond_offsets[o]].ffidx;
            tu_data->mi_to_tuffidx[idx] =     b+tu_data->bond_offsets[o];
            tu_data->mi_to_R[3*idx]   = map_to_fine(tu_ff[b+tu_data->bond_offsets[o]].R[0],
                                                        model->nk[0]);
            tu_data->mi_to_R[3*idx+1] = map_to_fine(tu_ff[b+tu_data->bond_offsets[o]].R[1],
                                                        model->nk[1]);
            tu_data->mi_to_R[3*idx+2] = map_to_fine(tu_ff[b+tu_data->bond_offsets[o]].R[2],
                                                        model->nk[2]);
            idx +=1;
        }else{
            tu_data->ob_to_orbff[b + tu_data->n_bonds * o] = -1;
        }
    }

    for(index_t o = 0; o < model->n_orb; ++o)
        assert(tu_data->mi_to_ofrom[o] == tu_data->mi_to_oto[o]);
    assert(n_tu_ff == idx);
}

static inline double claV4d_norm1( claV4d_t v ) {
    double res = 0.0;
    for (short i=0; i<4; ++i) res += fabs(v.v[i]);
    return res;
}

static inline int contains_claV4d_t( const claV4d_t* ary, index_t sz, claV4d_t test ) {
    int contains = 0;
    for (index_t i=0; i<sz; ++i) if (claV4d_norm1(claV4d_sub(ary[i], test)) < 1e-3) {
        contains = 1;
        break;
    }
    return contains;
}

void init_ff_struct(const double max_dist, diverge_model_t* model) {
    bool iobi_setup = false;
    if (qnan_isnan(max_dist) && model->n_tu_ff == 0) {
        uint32_t payload = qnan_get(max_dist);
        mpi_vrb_printf( "found NAN for max_dist, read payload '%u'\n", payload );
        if (payload == IOBI_FORMFACTOR_MAGIC_NUMBER) {
            mpi_log_printf( "dispatching to IOBI FRG\n" );
            iobi_setup = true;
        } else {
            mpi_err_printf( "invalid payload '%u', anyways doing IOBI\n", payload );
            // we could do different special cases here
            iobi_setup = true;
        }
        if (iobi_setup) {
            model->n_tu_ff = model->n_orb;
            model->tu_ff = (tu_formfactor_t*) calloc(model->n_tu_ff, sizeof(tu_formfactor_t));
            for (index_t f=0; f<model->n_tu_ff; ++f)
                model->tu_ff[f].ofrom = model->tu_ff[f].oto = f;
        }
    }

    if (model->n_tu_ff != 0) {
        if (!iobi_setup)
            mpi_wrn_printf("do not autogenerate bonds (user orbff)\n");
        sort_ff_struct(model->tu_ff, model->n_tu_ff);
        fill_existing_ff(model);
        init_bond_descriptor(model);
        return;
    }

    mpi_vrb_printf("autogenerate bonds (orbff)\n");
    int periodic_iter[3];
    periodic_iter[0] = model->internals->periodic_direction[0] ? 3:0;
    periodic_iter[1] = model->internals->periodic_direction[1] ? 3:0;
    periodic_iter[2] = model->internals->periodic_direction[2] ? 3:0;
    tu_data_t* tu_data = (tu_data_t*)model->internals->tu_data;
    index_t n_orb = model->n_orb;

    VECTOR_INIT( index_t, orbcomb );
    VECTOR_INIT( claV3i_t, real_ff );
    VECTOR_INIT( double, dist_from );

    claV3i_t delta;
    claM3d_t UCV = claM3d_map(model->lattice[0]);

    for (index_t sfrom=0; sfrom<n_orb; ++sfrom) {
        claV3d_t _from = claV3d_map(model->positions[sfrom]);
        for (index_t sto=0; sto<n_orb; ++sto) {
            VECTOR_INIT( claV4d_t, bond_lookup );
            for (index_t uc_1=0; uc_1<model->nk[0]; ++uc_1)
            for (index_t uc_2=0; uc_2<model->nk[1]; ++uc_2)
            for (index_t uc_3=0; uc_3<model->nk[2]; ++uc_3)  {
                claV3d_t to = claV3d_map(model->positions[sto]);

                claV3d_t from = claV3d_add( _from, claM3d_vecmat( (claV3d_t){{uc_1,uc_2,uc_3}}, UCV ) );
                claV3d_t dist = claV3d_sub( to, from );

                delta.v[0] = uc_1, delta.v[1] = uc_2, delta.v[2] = uc_3;

                for (int in=-periodic_iter[0]; in<=periodic_iter[0]; in++)
                for (int im=-periodic_iter[1]; im<=periodic_iter[1]; im++)
                for (int iq=-periodic_iter[2]; iq<=periodic_iter[2]; iq++) {
                    claV3d_t trial = claV3d_sub(to, from);
                    trial = claV3d_add( trial, claM3d_vecmat( (claV3d_t){{
                                in*model->nk[0], im*model->nk[1], iq*model->nk[2] }}, UCV ) );
                    if (claV3d_norm(trial) < claV3d_norm(dist)+1e-12)
                        dist = trial;
                }

                claV4d_t helper;
                claV3d_rmap( dist, helper.v );
                helper.v[3] = sto;

                if (claV3d_norm(dist) < max_dist+1e-10 && !contains_claV4d_t(bond_lookup,bond_lookup_sz,helper)) {
                    #pragma omp critical(write_into)
                    {
                        VECTOR_PUSH_BACK( bond_lookup, helper );
                        VECTOR_PUSH_BACK( real_ff, delta );
                        VECTOR_PUSH_BACK( orbcomb, sfrom );
                        VECTOR_PUSH_BACK( orbcomb, sto );
                        VECTOR_PUSH_BACK( dist_from, claV3d_norm(dist) );
                    }
                }
            }
            VECTOR_FREE( bond_lookup );
        }
    }

    model->n_tu_ff = real_ff_sz;
    model->tu_ff = calloc(real_ff_sz,sizeof(tu_formfactor_t));
    tu_formfactor_t* tu_ff = model->tu_ff;
    for (index_t i=0; i<model->n_tu_ff; ++i) {
        tu_ff[i].ofrom = orbcomb[2*i+0];
        tu_ff[i].oto   = orbcomb[2*i+1];
        assert(orbcomb[2*i+0] < n_orb);
        assert(orbcomb[2*i+1] < n_orb);
        tu_ff[i].R[0] = real_ff[i].v[0];
        tu_ff[i].R[1] = real_ff[i].v[1];
        tu_ff[i].R[2] = real_ff[i].v[2];
        tu_ff[i].d = dist_from[i];
    }
    bool refinement_allowed = true;
    for (index_t i=0; i<model->n_tu_ff; ++i) {
        claV3d_t from = claM3d_vecmat( (claV3d_t){{ map_to_fine(real_ff[i].v[0],model->nk[0]),
                                                    map_to_fine(real_ff[i].v[1],model->nk[1]),
                                                    map_to_fine(real_ff[i].v[2],model->nk[2]) }}, UCV );
        for (int in=-periodic_iter[0]; in<=periodic_iter[0]; in++)
        for (int im=-periodic_iter[1]; im<=periodic_iter[1]; im++)
        for (int iq=-periodic_iter[2]; iq<=periodic_iter[2]; iq++) {
            claV3d_t add = claM3d_vecmat( (claV3d_t){{ in*model->nk[0], im*model->nk[1], iq*model->nk[2] }}, UCV );
            claV3d_t trial = claV3d_add( from, add );
            if (fabs(claV3d_norm(from) - claV3d_norm(trial)) < 1e-8 && claV3d_norm(add) > 1e-8)
                refinement_allowed = false;
        }
    }

    if(!refinement_allowed && tu_data->nkf > 1)
        mpi_wrn_printf("too many ffs: refinement breaks symmetry\n");

    sort_ff_struct(model->tu_ff, model->n_tu_ff);
    fill_existing_ff(model);
    init_bond_descriptor(model);
    mpi_log_printf("generate %d ff-bonds with max. %d bonds per site\n", real_ff_sz,tu_data->n_bonds);

    VECTOR_FREE( dist_from );
    VECTOR_FREE( real_ff );
    VECTOR_FREE( orbcomb );
}

static index_t generate_unique_obff(diverge_model_t* model) {
    tu_data_t* tu_data = (tu_data_t*)(model->internals->tu_data);
    index_t n_orb = model->n_orb;
    index_t* bond_sizes = tu_data->bond_sizes;
    index_t* bond_offsets = tu_data->bond_offsets;
    tu_formfactor_t* tu_ff = model->tu_ff;

    index_t num_unique_pairs = 0;
    for (index_t o=0; o<n_orb; ++o) {
        VECTOR_INIT( index_t, have );
        for (index_t b=0; b<bond_sizes[o]; ++b) {
            index_t site = tu_ff[bond_offsets[o]+b].oto;
            index_t idx = -1;
            for (index_t h=0; h<have_sz; ++h) if (have[h] == site) {
                idx = h;
                break;
            }
            if (idx == -1) {
                num_unique_pairs++;
                VECTOR_PUSH_BACK( have, site );
            }
        }
        VECTOR_FREE( have );
    }
    return num_unique_pairs;
}

static inline index_t wrap_on_fine(index_t val, index_t ncf) {
    if (labs(val - ncf) < (ncf / 2) + 1) val = val - ncf;
    return val;
}

static void fill_unique_ff(diverge_model_t* model){
    tu_data_t* tu_data = (tu_data_t*)(model->internals->tu_data);
    // create array storing the info of all unique formfactor combinations in R
    index_t nff = tu_data->existing_ff_sz;
    tu_data->ff_idx_to_unique_ff_elem = calloc(POW2(nff),sizeof(index_t));

    for (index_t ff1=0; ff1<nff; ++ff1)
    for (index_t ff3=0; ff3<nff; ++ff3) {
        claV3i_t trial = claV3i_sub(tu_data->existing_ff_fine[ff1],tu_data->existing_ff_fine[ff3]);
        trial.v[0] = (trial.v[0]+4*model->nk[0]*model->nkf[0])%(model->nk[0]*model->nkf[0]);
        trial.v[1] = (trial.v[1]+4*model->nk[1]*model->nkf[1])%(model->nk[1]*model->nkf[1]);
        trial.v[2] = (trial.v[2]+4*model->nk[2]*model->nkf[2])%(model->nk[2]*model->nkf[2]);
        trial.v[0] = wrap_on_fine(trial.v[0],model->nk[0]*model->nkf[0]);
        trial.v[1] = wrap_on_fine(trial.v[1],model->nk[1]*model->nkf[1]);
        trial.v[2] = wrap_on_fine(trial.v[2],model->nk[2]*model->nkf[2]);
        index_t idx = 0;
        for (; idx<tu_data->unique_ff_combi_sz; idx++) {
            claV3i_t dd = claV3i_sub(tu_data->unique_ff_combi[idx], trial);
            if (claV3i_dot(dd,dd) == 0)
                break;
        }

        if (idx == tu_data->unique_ff_combi_sz)
            VECTOR_PUSH_BACK( tu_data->unique_ff_combi, trial );
        tu_data->ff_idx_to_unique_ff_elem[ff1+nff*ff3] = idx;
    }
}

static inline int compar_loopmap_idx( index_t a, index_t b, void* ctx ) {
    tu_data_t* tu_data = ctx;
    index_t mo1a = tu_data->unique_mi_l[a];
    index_t mo1b = tu_data->unique_mi_l[b];
    index_t mo3a = tu_data->unique_mi_r[a];
    index_t mo3b = tu_data->unique_mi_r[b];

    if (tu_data->mi_to_ofrom[mo1a] != tu_data->mi_to_ofrom[mo1b]) {
        return tu_data->mi_to_ofrom[mo1a] - tu_data->mi_to_ofrom[mo1b];
    } else if (tu_data->mi_to_ofrom[mo3a] != tu_data->mi_to_ofrom[mo3b]) {
        return tu_data->mi_to_ofrom[mo3a] - tu_data->mi_to_ofrom[mo3b];
    } else if (tu_data->mi_to_oto[mo1a] != tu_data->mi_to_oto[mo1b]) {
        return tu_data->mi_to_oto[mo1a] - tu_data->mi_to_oto[mo1b];
    } else {
        return tu_data->mi_to_oto[mo3a] - tu_data->mi_to_oto[mo3b];
    }
}
#undef MERGESORT_COMP_R
#define MERGESORT_COMP_R(a,b,ctx) compar_loopmap_idx(a,b,ctx)
MERGESORT_IMPL_R( mergesort_loopmap_idx, index_t )

void sort_loopmaps(diverge_model_t* model) {
    tu_data_t* tu_data = model->internals->tu_data;
    index_t n_unique_p = tu_data->unique_mi_l_sz;

    index_t* idx = calloc( n_unique_p, sizeof*idx );
    for (index_t d=0; d<n_unique_p; ++d) idx[d] = d;

    mergesort_loopmap_idx_r( idx, n_unique_p, tu_data, diverge_omp_num_threads(), 0 );

    index_t* uswl = calloc( n_unique_p, sizeof*uswl );
    index_t* uswr = calloc( n_unique_p, sizeof*uswr );
    index_t* uswoff = calloc( n_unique_p, sizeof*uswoff );
    index_t* uswlen = calloc( n_unique_p, sizeof*uswlen );
    index_t* uswmapl = calloc( tu_data->lmap_l_sz, sizeof*uswmapl );
    index_t* uswmapr = calloc( tu_data->lmap_l_sz, sizeof*uswmapr );
    for (index_t m=0; m<n_unique_p; ++m) {
        uswl[m] = tu_data->unique_mi_l[idx[m]];
        uswr[m] = tu_data->unique_mi_r[idx[m]];
        uswlen[m] = tu_data->lmap_len[idx[m]];
        if (m > 0)
            uswoff[m] = uswoff[m-1] + uswlen[m-1];
        for (index_t l=0; l<uswlen[m]; ++l) {
            uswmapl[l+uswoff[m]] = tu_data->lmap_l[tu_data->lmap_off[idx[m]]+l];
            uswmapr[l+uswoff[m]] = tu_data->lmap_r[tu_data->lmap_off[idx[m]]+l];
        }
    }
    free(tu_data->unique_mi_l); tu_data->unique_mi_l = uswl;
    free(tu_data->unique_mi_r); tu_data->unique_mi_r = uswr;
    free(tu_data->lmap_len); tu_data->lmap_len = uswlen;
    free(tu_data->lmap_off); tu_data->lmap_off = uswoff;
    free(tu_data->lmap_l); tu_data->lmap_l = uswmapl;
    free(tu_data->lmap_r); tu_data->lmap_r = uswmapr;

    free( idx );
}

typedef struct { VECTOR_DECL( claV2ll_t, v ) } vec_claV2ll_t;

void generate_reduced_loop_maps(diverge_model_t* model) {
    tu_data_t* tu_data = model->internals->tu_data;

    if (!tu_data->use_reduced_loop || tu_data->nktot==1) {
        mpi_log_printf("deactivate reduced TU loop algorithm\n");
        tu_data->use_reduced_loop = false;
        tu_data->ff_idx_to_unique_ff_elem = calloc(1,sizeof(index_t));
        return;
    }

    index_t n_orb = model->n_orb;
    index_t* mi_to_ofrom = tu_data->mi_to_ofrom;
    index_t* mi_to_oto = tu_data->mi_to_oto;
    index_t* mi_to_ffidx = tu_data->mi_to_ffidx;
    fill_unique_ff(model);
    index_t num_unique_pairs = generate_unique_obff(model);

    if (num_unique_pairs > (index_t)(0.7*tu_data->n_orbff)) {
        tu_data->use_reduced_loop = false;
        mpi_vrb_printf("speedup impossible, no orbff maps\n");
        return ;
    }
    mpi_vrb_printf("%d unique orbff pairs, gen maps…\n", num_unique_pairs);
    VECTOR_INIT( vec_claV2ll_t, _lmap );
    index_t nff = tu_data->existing_ff_sz;
    // iterate over all unique pair combinations
    xbtree_t* index_to_reduced = xbtree_init( MAX(tu_data->n_orbff,1024) );
    index_t curr_idx = 0;
    for (index_t m1 = 0; m1<tu_data->n_orbff; ++m1)
    for (index_t m3 = 0; m3<tu_data->n_orbff; ++m3) {
        index_t fidx1 = mi_to_ffidx[m1];
        index_t fidx3 = mi_to_ffidx[m3];
        index_t trial = IDX5(tu_data->ff_idx_to_unique_ff_elem[fidx1+nff*fidx3],
                          mi_to_oto[m3],mi_to_ofrom[m3],mi_to_oto[m1],mi_to_ofrom[m1],
                          n_orb,n_orb,n_orb,n_orb);
        btree_t* found = xbtree_find( index_to_reduced, trial );
        if (!found) {
            btree_t* node = xbtree_insert( index_to_reduced, trial );
            node->val = curr_idx++;

            claV2ll_t l = {{m1,m3}};
            vec_claV2ll_t b = {0};
            VECTOR_PUSH_BACK( b.v, l );
            VECTOR_PUSH_BACK( _lmap, b );
        } else {
            claV2ll_t l = {{m1,m3}};
            VECTOR_PUSH_BACK( _lmap[found->val].v, l );
        }
    }

    VECTOR_RESIZE( tu_data->unique_mi_l, _lmap_sz );
    VECTOR_RESIZE( tu_data->unique_mi_r, _lmap_sz );
    VECTOR_RESIZE( tu_data->lmap_off, _lmap_sz );
    VECTOR_RESIZE( tu_data->lmap_len, _lmap_sz );
    tu_data->lmap_off[0] = 0;
    VECTOR_RESERVE( tu_data->lmap_l, POW2(tu_data->n_orbff) );
    VECTOR_RESERVE( tu_data->lmap_r, POW2(tu_data->n_orbff) );
    for (index_t i=0; i<_lmap_sz; ++i) {
        tu_data->unique_mi_l[i] = _lmap[i].v[0].v[0];
        tu_data->unique_mi_r[i] = _lmap[i].v[0].v[1];
        for (index_t j=0; j<_lmap[i].v_sz; ++j) {
            VECTOR_PUSH_BACK( tu_data->lmap_l, _lmap[i].v[j].v[0] );
            VECTOR_PUSH_BACK( tu_data->lmap_r, _lmap[i].v[j].v[1] );
        }
        tu_data->lmap_len[i] = _lmap[i].v_sz;
        if (i>0)
            tu_data->lmap_off[i] = tu_data->lmap_off[i-1] + tu_data->lmap_len[i-1];
    }
    VECTOR_SHRINK_TO_FIT( tu_data->lmap_l );
    VECTOR_SHRINK_TO_FIT( tu_data->lmap_r );
    sort_loopmaps(model);
    mpi_vrb_printf("#unique orbff=%li (with %li unique ff from %li)\n",
                tu_data->unique_mi_l_sz, tu_data->unique_ff_combi_sz, POW2(tu_data->n_orbff));

    for (index_t i=0; i<_lmap_sz; ++i) VECTOR_FREE( _lmap[i].v );
    VECTOR_FREE( _lmap );
    xbtree_free( index_to_reduced );
}

void init_MPI_dist(diverge_model_t* model) {
    tu_data_t* tu_data = (tu_data_t*)(model->internals->tu_data);
    int rank = diverge_mpi_comm_rank();
    int nranks = diverge_mpi_comm_size();
    // distribute kpoints
    index_t to_be_distributed = tu_data->nkibz;
    if (to_be_distributed < nranks)
        mpi_err_printf("nkibz (%li) < #ranks (%i). reduce #ranks!\n", to_be_distributed, nranks);

    index_t momenta_per_rank = to_be_distributed/nranks;
    index_t remainder = to_be_distributed%nranks;

    if(rank < remainder)
    {
        tu_data->my_nk = momenta_per_rank+1;
        tu_data->my_nk_off = rank*(momenta_per_rank+1);
    }else{
        tu_data->my_nk = momenta_per_rank;
        tu_data->my_nk_off = remainder * (momenta_per_rank+1)
                           + momenta_per_rank * (rank-remainder);
    }
    diverge_mpi_barrier();
    mpi_log_printf_all( "rank=%d mynk=%d nkibz=%d\n", rank, tu_data->my_nk, tu_data->nkibz);
}
