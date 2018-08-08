=====
pywsi
=====


.. image:: https://travis-ci.com/saketkc/pywsi.svg?token=GsuWFnsdqcXUSp8vzLip&branch=master
        :target: https://travis-ci.com/saketkc/pywsi

.. image:: ./logo/virchow.png
        :target: ./logo/virchow.png

* Free software: BSD license
* Documentation: https://www.saket-choudhary.me/pywsi/


Features
--------

See the Notebook_.

Training InceptionV4 on Tumor/Normal patches
--------------------------------------------

We currently rely on InceptionV4_ model for training. It is one of the 
deepest and most sophesticated models available. Another model we would ideally
like to explore is Inception-Resnet, but later.


Step 1. Create tissue masks
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  pywsi create-tissue-masks --indir /CAMELYON16/testing/images/ \
  --level 5 --savedir /CAMELYON16/testing/tissue_masks


Step 2. Create annotation masks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pywsi create-annotation-masks --indir /CAMELYON16/testing/images/ \
   --level 5 --savedir /CAMELYON16/testing/annotation_masks \
   --jsondir /CAMELYON16/testing/lesion_annotations_json

Step 3A. Extract tumor patches
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pywsi extract-tumor-patches --indir /CAMELYON16/testing/images/ \
   --annmaskdir /CAMELYON16/testing/annotation_masks \
   --tismaskdir /CAMELYON16/testing/tissue_masks \
   --level 5 --savedir /CAMELYON16/testing/extracted_tumor_patches


Step 3B. Extract normal patches
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pywsi extract-normal-patches --indir /CAMELYON16/training/normal \
   --tismaskdir /CAMELYON16/training/tissue_masks --level 5 \
   --savedir /CAMELYON16/training/extracted_normal_patches


Dataset download
=================

Ftp_


.. _InceptionV4: https://arxiv.org/abs/1602.07261
.. _Notebook: notebooks/01.pywsi-demo.ipynb
.. _Ftp: ftp://parrot.genomics.cn/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/
