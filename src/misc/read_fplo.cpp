#include "read_fplo.h"
#include "../diverge_Eigen3.hpp"

#include <vector>
#include <array>
#include <fstream>
#include <string>
#include <iostream>
#include <variant>

using std::vector;
using std::array;
using std::string;

typedef enum {
    parser_state_matrix,
    parser_state_vectors,
    parser_state_integer,
    parser_state_integers,
    parser_state_hoppings,

    parser_state_none,
    parser_state_eof,
} parser_state_e;

/* typedef union { */
/*     Mat3d* matrix; */
/*     vector<Vec3d>* vectors; */
/*     index_t* integer; */
/*     array<index_t,2>* integers; */
/*     vector<array<double,5>>* hoppings; */
/* } ptr_union_t; */

/* class parser_object_u { */
/*     public: */
/*         ptr_union_t p; */
/*         parser_object_u(parser_state_e state) { */
/*             switch (state) { */
/*                 case parser_state_matrix: p.matrix = new Mat3d(); break; */
/*                 case parser_state_vectors: p.vectors = new vector<Vec3d>(); break; */
/*                 case parser_state_integer: p.integer = new index_t; break; */
/*                 case parser_state_integers: p.integers = new array<index_t,2>(); break; */
/*                 case parser_state_hoppings: p.hoppings = new vector<array<double,5>>(); break; */
/*                 default: break; */
/*             } */
/*             _state = state; */
/*         } */
/*         ~parser_object_u() { */
/*             switch (_state) { */
/*                 case parser_state_matrix: delete p.matrix; break; */
/*                 case parser_state_vectors: delete p.vectors; break; */
/*                 case parser_state_integer: delete p.integer; break; */
/*                 case parser_state_integers: delete p.integers; break; */
/*                 case parser_state_hoppings: delete p.hoppings; break; */
/*                 default: break; */
/*             } */
/*         } */
/*     private: */
/*         parser_state_e _state; */
/* }; */

parser_state_e next_parser_state( const char* line ) {
    if (!strncmp(line, "lattice_vectors:", 16)) return parser_state_matrix;
    if (!strncmp(line, "wancenters:", 11)) return parser_state_vectors;
    if (!strncmp(line, "nwan:", 5)) return parser_state_integer;
    if (!strncmp(line, "Tij, Hij:", 9)) return parser_state_integers;
    return parser_state_none;
}

typedef std::variant<
    Mat3d,
    vector<Vec3d>,
    index_t,
    array<index_t,2>,
    vector<array<double,5>>
> parser_object_u;

parser_object_u get_state_data( std::ifstream& file, parser_state_e& state ) {
    string line_;
    parser_state_e state_in = state;

    index_t row = 0;

    Mat3d state_matrix = Mat3d::Zero();
    vector<Vec3d> state_vectors;
    index_t state_integer = 0;
    array<index_t,2> state_integers;
    vector<array<double,5>> state_hoppings;

    union {
        double doubles[5];
        index_t integers[2];
    };

    parser_object_u r;

    while (std::getline(file, line_)) {
        const char* line = line_.c_str();

        bool success = false;

        switch (state_in) {
            case parser_state_matrix:
            case parser_state_vectors:
                success = (sscanf( line, " %lf %lf %lf", &doubles[0], &doubles[1], &doubles[2] ) == 3);
                break;
            case parser_state_integer:
                success = (sscanf( line, " %ld", &integers[0] ) == 1);
                break;
            case parser_state_integers:
                success = (sscanf( line, " %ld %ld", &integers[0], &integers[1] ) == 2);
                break;
            case parser_state_hoppings:
                success = (sscanf( line, " %lf %lf %lf %lf %lf", &doubles[0], &doubles[1], &doubles[2], &doubles[3], &doubles[4] ) == 5);
                break;
            default:
                break;
        }

        if (success) {
            switch (state_in) {
                case parser_state_matrix:
                    if (row < 3) {
                        for (int i=0; i<3; ++i) state_matrix(row,i) = doubles[i];
                    } else {
                        success = false;
                    }
                    break;
                case parser_state_vectors:
                    state_vectors.push_back( Map<Vec3d>(doubles) );
                    break;
                case parser_state_integer:
                    state_integer = integers[0];
                    break;
                case parser_state_integers:
                    state_integers[0] = integers[0];
                    state_integers[1] = integers[1];
                    success = false;
                    break;
                case parser_state_hoppings:
                    state_hoppings.push_back( {doubles[0], doubles[1], doubles[2], doubles[3], doubles[4]} );
                    break;
                default:
                    break;
            }
        }

        if (!success) {
            state = (state_in == parser_state_integers) ? parser_state_hoppings : next_parser_state( line );
            goto return_state;
        }

        row++;
    }

    state = parser_state_eof;

return_state:
    switch (state_in) {
        case parser_state_matrix: r = state_matrix; break;
        case parser_state_vectors: r = state_vectors; break;
        case parser_state_integer: r = state_integer; break;
        case parser_state_integers: r = state_integers; break;
        case parser_state_hoppings: r = state_hoppings; break;
        default: break;
    }
    return r;
}

// TODO cannot treat non-SU2 at the moment
rs_hopping_t* diverge_read_fplo( const char* fname, index_t* n_hop, index_t* n_orb ) {

    Mat3d lat;
    Mat3d lat_inv;
    vector<Vec3d> positions;
    vector<rs_hopping_t> hop;
    index_t nwan = 0;

    std::ifstream f(fname);
    if (!f) {
        mpi_err_printf( "could not open '%s' for reading\n", fname );
        return NULL;
    }

    parser_state_e state = parser_state_none;
    array<index_t,2> o12;

    while (state != parser_state_eof) {
        parser_state_e prevstate = state;
        parser_object_u data = get_state_data(f, state);

        Vec3d Rvec = Vec3d::Zero();
        rs_hopping_t h;
        switch (prevstate) {
            case parser_state_matrix:
                lat = std::get<Mat3d>(data);
                lat_inv = lat.inverse();
                break;
            case parser_state_vectors:
                positions = std::get<vector<Vec3d>>(data);
                break;
            case parser_state_integer:
                nwan = std::get<index_t>(data);
                break;
            case parser_state_integers:
                o12 = std::get<array<index_t,2>>(data);
                break;
            case parser_state_hoppings:
                for (const auto& dbls: std::get<vector<array<double,5>>>(data)) {
                    Rvec = lat_inv * Map<const Vec3d>( dbls.data() );
                    for (int i=0; i<3; ++i) h.R[i] = lround(Rvec(i));
                    h.o1 = o12[1] - 1;
                    h.o2 = o12[0] - 1;
                    h.t = complex128_t(dbls[3],dbls[4]);
                    hop.push_back( h );
                }
                break;
            default:
                break;
        }
    }

    /* for (const auto& p: positions) */
    /*     std::cout << p.transpose() << std::endl; */

    if (n_hop) *n_hop = hop.size();
    if (n_orb) *n_orb = nwan;

    rs_hopping_t* hop_ptr = (rs_hopping_t*)calloc( *n_hop, sizeof(rs_hopping_t) );
    memcpy( hop_ptr, hop.data(), sizeof(rs_hopping_t)*hop.size() );
    return hop_ptr;
}
