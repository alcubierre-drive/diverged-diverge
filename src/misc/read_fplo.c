/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */
#include "read_fplo.h"
#include "text_io.h"
#include "invert.h"
#include <assert.h>

#include "../diverge_model.h"
#define LINALG_NO_LAPACKE
#include "linalg.h"
LINALG_IMPLEMENT

typedef struct {
    rs_hopping_t* hop;
    index_t n_hop;

    index_t n_orb;
    index_t n_spin;

    double lattice[3][3];
    double* pos;
} fplo_hamiltonian_t;

static fplo_hamiltonian_t* fplo_hamiltonian_read( const char* fplo_file );
static void fplo_hamiltonian_free( fplo_hamiltonian_t* h );

diverge_model_t* diverge_read_fplo( const char* fplo_file ) {
    fplo_hamiltonian_t* h = fplo_hamiltonian_read( fplo_file );
    if (!h) return NULL;
    diverge_model_t* m = diverge_model_init();
    m->hop = h->hop;
    m->n_hop = h->n_hop;
    m->n_orb = h->n_orb;
    m->n_spin = h->n_spin;
    memcpy( m->lattice[0], h->lattice[0], sizeof(double)*9 );
    memcpy( m->positions[0], h->pos, sizeof(double)*h->n_orb*3 );

    h->hop = NULL;
    fplo_hamiltonian_free( h );
    return m;
}

typedef struct {
    double R[3];
    complex128_t t;
} fplo_single_hopping_t;

typedef struct fplo_hop_element_t fplo_hop_element_t;
typedef struct fplo_hop_element_t {
    VECTOR_DECL(fplo_single_hopping_t, hops)

    index_t o1, o2;
    index_t spin;

    fplo_hop_element_t* next;
} fplo_hop_element_t;

static fplo_hop_element_t* fplo_hop_element_init( void );
static void fplo_hop_element_free( fplo_hop_element_t* e );
static fplo_hop_element_t* fplo_hop_element_append( fplo_hop_element_t* head, fplo_hop_element_t* elm ) {
    if (!head) return (head = elm);
    fplo_hop_element_t* tmp = head;
    while (tmp->next != NULL) tmp = tmp->next;
    tmp->next = elm;
    return head;
}

typedef struct {
    double lattice_vectors[3][3];
    VECTOR_DECL( claV3d_t, fplo_pos )
    fplo_hop_element_t* fplo_hops;
    index_t fplo_nwan;
    index_t fplo_nspin;

    int valid;
} fplo_intermediate_output_t;

static fplo_intermediate_output_t fplo_intermediate_read( const char* fplo_file );

static fplo_hop_element_t* fplo_hop_element_init( void ) {
    fplo_hop_element_t* e = calloc( 1, sizeof(fplo_hop_element_t) );
    VECTOR_RESERVE(e->hops, 8);
    e->o1 = e->o2 = e->spin = -1;
    return e;
}

static void fplo_hop_element_free( fplo_hop_element_t* e ) {
    VECTOR_FREE(e->hops);
    free( e );
}

static const int REGEX_MATCH_FPLO_LABEL_SUCCESS = 0,
                 REGEX_MATCH_FPLO_LABEL_NO_COLON = 1,
                 REGEX_MATCH_FPLO_LABEL_NO_TRAILING_SPACE = 2,
                 REGEX_MATCH_FPLO_LABEL_EMPTY_STRING = 4;

static inline bool is_label_char( const char x ) {
    const char label_ranges[][2] = { {'A','Z'}, {'a','z'}, {'0','9'}, {'_','_'}, {' ',' '}, {',',','} };
    const int n_label_ranges = sizeof label_ranges / 2;
    bool is_label_char = false;
    for (int r=0; r<n_label_ranges; ++r) {
        if ((x >= label_ranges[r][0]) && (x <= label_ranges[r][1])) {
            is_label_char = true;
            break;
        }
    }
    return is_label_char;
}

static inline bool is_space_char( const char x ) {
    const char space_chars[] = { ' ', '\t', '\r', '\n', '\v', '\f', };
    const int n_space_chars = sizeof space_chars;
    bool is_space = false;
    for (int s=0; s<n_space_chars; ++s) {
        if (x == space_chars[s]) {
            is_space = true;
            break;
        }
    }
    return is_space;
}

static inline int regex_match_fplo_label( const char* line ) {
    int parse_label = 1;
    size_t nchar = 0;
    for (const char* c=line; *c; c++) {
        nchar++;
        if (parse_label) {
            if (!is_label_char(*c)) {
                parse_label = 0;
                if (*c != ':') return REGEX_MATCH_FPLO_LABEL_NO_COLON;
            }
        } else {
            if (!is_space_char(*c)) return REGEX_MATCH_FPLO_LABEL_NO_TRAILING_SPACE;
        }
    }
    if (nchar == 0) return REGEX_MATCH_FPLO_LABEL_EMPTY_STRING;
    return REGEX_MATCH_FPLO_LABEL_SUCCESS;
}

static fplo_intermediate_output_t fplo_intermediate_read( const char* fplo_file ) {

    fplo_intermediate_output_t io = {.valid = 0};

    FILE* stream = fopen(fplo_file, "r");
    if (!stream) {
        mpi_err_printf( "could not open file '%s'\n", fplo_file );
        return io;
    }

    // output objects
    double lattice_vectors[3][3] = {0};
    fplo_hop_element_t* fplo_hops = NULL;
    VECTOR_INIT( claV3d_t, fplo_pos );
    index_t fplo_nwan = 0;
    index_t fplo_nspin = 0;

    // parser state
    index_t current_hopping_s = -1;
    fplo_hop_element_t* current_hopping = NULL;

    int parse_sect_hop = 0;
    int parse_sect_hop_next_line_set_orbitals = 0;

    int parse_sect_spin = 0;
    int parse_sect_spin_next_line_set_spin = 0;

    int parse_latvec = 0;
    int parse_latvec_i = 0;

    int parse_nwan = 0;
    int parse_centers = 0;

#define RESET_PARSE_STATE() { \
    parse_latvec = parse_nwan = parse_centers = parse_sect_hop = parse_sect_spin = 0; \
}

    index_t nread = 0;
    char line[DIVERGE_GETLINE_MAX] = {0};
    int linenum = 0;

    int errors = 0;

    while ((nread = diverge_getline(line, stream)) != -1) {
        linenum++;
        int res = regex_match_fplo_label(line);
        if (res == 0) {
            char str[DIVERGE_GETLINE_MAX] = {0};
            char *l = line, *s = str;
            while ((*s++ = *l++) != ':') {}
            *(s-1) = '\0';

            if (!strcmp(str, "lattice_vectors")) {
                RESET_PARSE_STATE();
                parse_latvec = 1;
            } else if (!strcmp(str, "nwan")) {
                RESET_PARSE_STATE();
                parse_nwan = 1;
            } else if (!strcmp(str, "wancenters")) {
                RESET_PARSE_STATE();
                parse_centers = 1;
            } else if (!strcmp(str, "spin")) {
                RESET_PARSE_STATE();
                // spin
                fplo_nspin++;
                parse_sect_spin = 1;
                parse_sect_spin_next_line_set_spin = 1;
            } else if (!strcmp(str, "end spin")) {
                parse_sect_spin = 0;
            } else if (!strcmp(str, "Tij, Hij")) {
                // Tij
                current_hopping = fplo_hop_element_init();
                current_hopping->spin = current_hopping_s;
                parse_sect_hop = 1;
                parse_sect_hop_next_line_set_orbitals = 1;
            } else if (!strcmp(str, "end Tij, Hij")) {
                fplo_hops = fplo_hop_element_append( fplo_hops, current_hopping );
                current_hopping = NULL;
                parse_sect_hop = 0;
            } else {
                RESET_PARSE_STATE();
            }
        } else {
            if (parse_latvec) {
                if (parse_latvec_i < 3) {
                    int nscan = sscanf(line, " %le %le %le",
                            &lattice_vectors[parse_latvec_i][0],
                            &lattice_vectors[parse_latvec_i][1],
                            &lattice_vectors[parse_latvec_i][2]);
                    parse_latvec_i++;
                    if (nscan != 3) {
                        mpi_err_printf( "could not read lattice_vectors (l.%i)\n", linenum );
                        errors++;
                    }
                }
            }
            if (parse_nwan) {
                int nscan = sscanf(line, " %ld", &fplo_nwan);
                if (nscan != 1) {
                    mpi_err_printf( "could not read nwan (l.%i)\n", linenum );
                    errors++;
                }
            }
            if (parse_centers) {
                claV3d_t p = {0};
                int nscan = sscanf(line, " %le %le %le", &p.v[0], &p.v[1], &p.v[2]);
                VECTOR_PUSH_BACK( fplo_pos, p );
                if (nscan != 3) {
                    mpi_err_printf( "could not read position (l.%i)\n", linenum );
                    errors++;
                }
            }
            if (parse_sect_spin && parse_sect_spin_next_line_set_spin) {
                int nscan = sscanf(line, " %ld", &current_hopping_s);
                if (nscan != 1) {
                    mpi_err_printf( "could not read spin (l.%i)\n", linenum );
                    errors++;
                }
                parse_sect_spin_next_line_set_spin = 0;
            }
            if (parse_sect_hop) {
                if (parse_sect_hop_next_line_set_orbitals) {
                    int nscan = sscanf(line, " %ld %ld", &current_hopping->o1, &current_hopping->o2);
                    if (nscan != 2) {
                        mpi_err_printf( "could not read orbital info for hop (l.%i)\n", linenum );
                        errors++;
                    }
                    parse_sect_hop_next_line_set_orbitals = 0;
                } else {
                    double dbls[5] = {0};
                    int nscan = sscanf(line, " %le %le %le %le %le", &dbls[0], &dbls[1], &dbls[2], &dbls[3], &dbls[4]);
                    if (nscan != 5) {
                        mpi_err_printf( "could not read hopping (l.%i)\n", linenum );
                        errors++;
                    }
                    VECTOR_PUSH_BACK(current_hopping->hops, (fplo_single_hopping_t){
                            .R = {dbls[0], dbls[1], dbls[2]}, .t = dbls[3] + I*dbls[4] });
                }
            }
        }
    }
    fclose(stream);

    io = (fplo_intermediate_output_t){.fplo_nwan = fplo_nwan,
                                      .fplo_hops = fplo_hops,
                                      .fplo_nspin = fplo_nspin,
                                      .fplo_pos = fplo_pos, .fplo_pos_sz = fplo_pos_sz, .fplo_pos_cap = fplo_pos_cap};
    memcpy(io.lattice_vectors[0], lattice_vectors[0], sizeof(double)*9);
    if (!errors) io.valid = 1;
    return io;
}

static fplo_hamiltonian_t* fplo_hamiltonian_read( const char* fplo_file ) {
    fplo_intermediate_output_t io = fplo_intermediate_read( fplo_file );
    if (!io.valid) {
        mpi_err_printf( "got invalid +hamdata, return NULL\n" );
        VECTOR_FREE(io.fplo_pos);
        for (fplo_hop_element_t* h=io.fplo_hops; h; ) {
            fplo_hop_element_t* x=h; h=h->next; fplo_hop_element_free(x);
        }
        return NULL;
    }
    fplo_hamiltonian_t* h = calloc( 1, sizeof(fplo_hamiltonian_t) );
    h->n_orb = io.fplo_nwan;
    h->n_spin = io.fplo_nspin;
    memcpy( h->lattice[0], io.lattice_vectors[0], sizeof(double)*9 );

    claM3d_t lattice = claM3d_map(h->lattice[0]);
    claM3d_t rlattice = {0};
    matrix_invert( lattice.m[0], 3, rlattice.m[0] );
    rlattice = claM3d_transpose(rlattice);

    h->pos = calloc( h->n_orb*3, sizeof(double) );
    assert( sizeof(claV3d_t) == (3*sizeof(double)) );
    if (h->n_orb != io.fplo_pos_sz) {
        mpi_err_printf( "#orb (%li) doesn't match #pos (%li)\n", h->n_orb, io.fplo_pos_sz );
        free( h );
        VECTOR_FREE(io.fplo_pos);
        for (fplo_hop_element_t* h=io.fplo_hops; h; ) {
            fplo_hop_element_t* x=h; h=h->next; fplo_hop_element_free(x);
        }
        return NULL;
    }
    memcpy( h->pos, io.fplo_pos, 3*sizeof(double) * h->n_orb );
    VECTOR_FREE(io.fplo_pos);

    for (fplo_hop_element_t* hop=io.fplo_hops; hop; hop = hop->next)
        h->n_hop += hop->hops_sz;
    h->hop = calloc(h->n_hop, sizeof(rs_hopping_t));
    h->n_hop = 0;
    for (fplo_hop_element_t* hop=io.fplo_hops; hop; ) {
        index_t o1 = hop->o1-1,
                o2 = hop->o2-1;
        for (index_t i=0; i<hop->hops_sz; ++i) {
            claV3d_t Rvec = claV3d_add( claV3d_map(hop->hops[i].R),
                    claV3d_sub(claV3d_map(h->pos+o1*3), claV3d_map(h->pos+o2*3)) );
            claV3d_t Rivec = claM3d_matvec( rlattice, Rvec );
            index_t Ri[3] = {0};
            for (int x=0; x<3; ++x) {
                Ri[x] = lround(Rivec.v[x]);
            }
            rs_hopping_t rhop = {
                .R={Ri[0], Ri[1], Ri[2]}, .o1=o1, .o2=o2,
                .s1=hop->spin-1, .s2=hop->spin-1,
                .t=hop->hops[i].t
            };
            h->hop[h->n_hop++] = rhop;
        }
        fplo_hop_element_t* x = hop; hop = hop->next;
        fplo_hop_element_free( x );
    }
    return h;
}

static void fplo_hamiltonian_free( fplo_hamiltonian_t* h ) {
    if (h) {
        if (h->hop) free( h->hop );
        if (h->pos) free( h->pos );
        free( h );
    }
}

