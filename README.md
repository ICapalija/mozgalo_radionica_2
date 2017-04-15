# Mozgalo radionica 2

[![N|Solid](http://www.netokracija.com/wp-content/uploads/2016/09/micro-blink-logo.png)](https://microblink.com/en)

U repozitoriju se nalazi programski kod koji se koristio na drugoj radionici natjecanja Mozgalo. Primjeri su implementirani u razvojnom okruženju [TensorFlow](https://www.tensorflow.org/). Sva pitanja vezana uz [TensorFlow](https://www.tensorflow.org/) možete pronaći na službenoj stranici.

#### Unsupervised learning
##### Train

  - Naredba za pokretanje treniranja: ``` python train.py ```
  -- Naučeni model će biti spremljen u folderu "models"
##### Extract embeddings
  - Naredba za izvlaćenje vektora: ``` python extract_embeddings.py --model <model_path>```
  -- Ukoliko želite vidjeti kako izgleda rekonstruirana slika pokrenite naredbu s  ```--vis```
  - Naredba za pokretanje TensorBoard vizualizacije: ``` tensorboard --logdir tensorboard/test ```

#### Transfer learning
##### Extract embeddings
  - Naredba za izvlaćenje vektora: ``` python extract_embeddings.py --model <model_path>```
  -- Ukoliko želite vidjeti kako izgleda ulazna slika pokrenite naredbu s ```--vis```
  -- Na radionici se koristio [Inception](https://storage.googleapis.com/download.tensorflow.org/models/inception_dec_2015.zip) model
  - Naredba za pokretanje TensorBoard vizualizacije: ``` tensorboard --logdir tensorboard/test ```

