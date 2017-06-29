pyltr
=====

|pypi version| |Build status|

.. |pypi version| image:: https://img.shields.io/pypi/v/pyltr.svg
   :target: https://pypi.python.org/pypi/pyltr
.. |Build status| image:: https://secure.travis-ci.org/jma127/pyltr.svg
   :target: http://travis-ci.org/jma127/pyltr

pyltr is a Python learning-to-rank toolkit with ranking models, evaluation
metrics, data wrangling helpers, and more.

This software is licensed under the BSD 3-clause license (see ``LICENSE.txt``).

The author may be contacted at ``ma127jerry <@t> gmail`` with general
feedback, questions, or bug reports.


Example
=======

Import pyltr::

    import pyltr

Import a `LETOR
<http://research.microsoft.com/en-us/um/beijing/projects/letor/>`_ dataset
(e.g. `MQ2007
<http://research.microsoft.com/en-us/um/beijing/projects/letor/LETOR4.0/Data/MQ2007.rar>`_
)::

    with open('train.txt') as trainfile, \
            open('vali.txt') as valifile, \
            open('test.txt') as evalfile:
        TX, Ty, Tqids, _ = pyltr.data.letor.read_dataset(trainfile)
        VX, Vy, Vqids, _ = pyltr.data.letor.read_dataset(valifile)
        EX, Ey, Eqids, _ = pyltr.data.letor.read_dataset(evalfile)

Train a `LambdaMART
<http://research.microsoft.com/pubs/132652/MSR-TR-2010-82.pdf>`_ model, using
validation set for early stopping and trimming::

    metric = pyltr.metrics.NDCG(k=10)

    # Only needed if you want to perform validation (early stopping & trimming)
    monitor = pyltr.models.monitors.ValidationMonitor(
        VX, Vy, Vqids, metric=metric, stop_after=250)

    model = pyltr.models.LambdaMART(
        metric=metric,
        n_estimators=1000,
        learning_rate=0.02,
        max_features=0.5,
        query_subsample=0.5,
        max_leaf_nodes=10,
        min_samples_leaf=64,
        verbose=1,
    )

    model.fit(TX, Ty, Tqids, monitor=monitor)

Evaluate model on test data::

    Epred = model.predict(EX)
    print 'Random ranking:', metric.calc_mean_random(Eqids, Ey)
    print 'Our model:', metric.calc_mean(Eqids, Ey, Epred)


Features
========

Below are some of the features currently implemented in pyltr.


Models
------
* LambdaMART (``pyltr.models.LambdaMART``)

  - Validation & early stopping

  - Query subsampling


Metrics
-------
* (N)DCG (``pyltr.metrics.DCG``, ``pyltr.metrics.NDCG``)

  - pow2 and identity gain functions

* ERR (``pyltr.metrics.ERR``)

  - pow2 and identity gain functions

* (M)AP (``pyltr.metrics.AP``)

* Kendall's Tau (``pyltr.metrics.KendallTau``)

* AUC-ROC -- Area under the ROC curve (``pyltr.metrics.AUCROC``)


Data Wrangling
--------------
* Data loaders (e.g. ``pyltr.data.letor.read``)

* Query groupers and validators
  (``pyltr.util.group.check_qids``, ``pyltr.util.group.get_groups``)


Running Tests
=============

Use the ``run_tests.sh`` script to run all unit tests.


Building Docs
=============

``cd`` into the ``docs/`` directory and run ``make html``. Docs are generated
in the ``docs/_build`` directory.


Contributing
============

Quality contributions or bugfixes are gratefully accepted. When submitting a
pull request, please update ``AUTHOR.txt`` so you can be recognized for your
work :).

By submitting a Github pull request, you consent to have your submitted code
released under the terms of the project's license (see ``LICENSE.txt``).
