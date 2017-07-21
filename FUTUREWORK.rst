Future Work
===========

Right now, pyltr is still in the "toy project" stage. Specifically, it is
rather slow and limited in its modeling capabilities. To the best of my
knowledge, most industry LTR practitioners today still use RankLib, which is
unfortunate since RankLib's only interfaces are via Java and command line --
both suboptimal for research and prototyping.

Based on user feedback, it seems that the primary appeal of pyltr is its
facilitation of a lightweight interactive research workflow. Expanding pyltr's
scope to be as comprehensive as RankLib will fill a significant gap in LTR, as
it will consolidate the most researched and used models/metrics into a single
package that can take advantage of the rich Python data science ecosystem.

As such, my long-term vision for this project is to make it a first-class
competitor to RankLib. Ideally, it would become the "go-to" LTR library for
both research and production training.


Process
=======

The realization of the above will most likely involve:

- a near-total rewrite to improve code structure
- significant interface changes
- cythonized/JITted critical-path code
- use of a more optimized tree engine e.g. LightGBM
- model zoo (most of the Ranklib models + potentially more)
- command-line interface (though RankLib cmdline compatibility is *not* a goal)
- support for only Python 3 going forward


v0 Incompatibility
==================

The project will be bumped to v1 upon completion of the above, and it is
probable that code using pyltr v0 will no longer work with v1 and above. Legacy
releases will still be hosted on Github and pypi.


Feedback
========

If you have any feedback or would like to contribute, feel free to email me
(ma127jerry @t gmail)!
