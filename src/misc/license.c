/* Copyright (C) 2026 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "license.h"
#include "mpi_log.h"

static const char license_info[] =
"divERGe implements various ERG examples\n"
"Copyright (C) 2024 J.B.Profe, L.Klebl\n"
"\n"
"This program is free software: you can redistribute it and/or modify\n"
"it under the terms of the GNU General Public License as published by\n"
"the Free Software Foundation, either version 3 of the License, or (at\n"
"your option) any later version.\n"
"\n"
"This program is distributed in the hope that it will be useful, but\n"
"WITHOUT ANY WARRANTY; without even the implied warranty of\n"
"MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU\n"
"General Public License for more details.\n"
"\n"
"You should have received a copy of the GNU General Public License\n"
"along with this program.  If not, see https://www.gnu.org/licenses/.\n";

const char* diverge_license( void ) {
    return license_info;
}

void diverge_license_print( void ) {
    mpi_printf( "%s", diverge_license() );
}
