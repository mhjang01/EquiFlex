// Force-included by scripts/build_smina.sh: lets the smina sources, written
// against Open Babel 2.x and an older Boost, compile against Open Babel 3 and
// Boost >= 1.83 without touching the smina tree.
#pragma once
#define BOOST_TIMER_ENABLE_DEPRECATED
#ifdef __cplusplus
#include <stack>
// Open Babel 3 no longer pulls these in through mol.h.
#include <openbabel/mol.h>
#include <openbabel/atom.h>
#include <openbabel/bond.h>
#include <openbabel/residue.h>
#include <openbabel/generic.h>
#include <openbabel/obiter.h>
#include <openbabel/elements.h>
#include <boost/serialization/optional.hpp>

// Renamed in Open Babel 3. Defined after the Open Babel headers, so only
// smina's own call sites are rewritten.
#define GetBO GetBondOrder
#define GetHvyValence GetHvyDegree

// Boost 1.83 version<optional<T>> lacks the mpl type/tag members the archive
// checks require. optional<T> is object_serializable, so the version is 0.
namespace boost { namespace serialization {
#define SMINA_OPTIONAL_VERSION(T) template<> struct version<boost::optional<T> > { \
    typedef mpl::int_<0> type; typedef mpl::integral_c_tag tag; \
    BOOST_STATIC_CONSTANT(int, value = 0); };
SMINA_OPTIONAL_VERSION(unsigned long)
SMINA_OPTIONAL_VERSION(unsigned short)
#undef SMINA_OPTIONAL_VERSION
}}
#endif
