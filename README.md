# Contact Localization

Given the object’s mesh and different sensory observations of the contact position (visual images, impact sounds, or tactile readings), this task aims to predict the vertex coordinate of the surface location on the mesh where the contact happens.

More formally, the task can be defined as follows: given a visual patch image V  (i.e., a visual image near the object’s surface) and/or a tactile reading T and/or an impact sound S, and the shape of the object P (represented by a point cloud), the model needs to localize the contact position C on the point cloud.

## Usage

#### Training & Evaluation

Start the training process, and test the best model on test-set after training:

```sh
python main.py --batch_size 8 --modality_list vision touch audio \
               --model CLR --weight_decay 1e-2 --lr 5e-4 \
			   --exp CLR_vision_touch_audio
```

Evaluate the best model in *CLR_vision_touch_audio*:

```sh
python main.py --batch_size 8 --modality_list vision touch audio \
			   --model CLR --weight_decay 1e-2 --lr 5e-4 \
			   --exp CLR_vision_touch_audio \
			   --eval
```

#### Add your own model

To train and test your new model on ObjectFolder Contact Localiazation Benchmark, you only need to modify several files in *models*, you may follow these simple steps.

1. Create new model directory

    ```sh
    mkdir models/my_model
    ```

2. Design new model

    ```sh
    cd models/my_model
    touch my_model.py
    ```

3. Build the new model and its optimizer

    Add the following code into *models/build.py*:

    ```python
    elif args.model == 'my_model':
        from my_model import my_model
        model = my_model.my_model(args)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ```

4. Add the new model into the pipeline

    Once the new model is built, it can be trained and evaluated similarly:

    ```sh
    python main.py --modality_list vision touch audio \
    							 --model my_model \
    							 --exp my_model
    ```

## Results on ObjectFolder Contact Localization Benchmark

In our experiments, we manually choose 50 objects with rich surface features from the dataset, and sample 1, 000 contacts from each object. The sampling strategy is based on the surface curvature. We assume that the curvature of each vertex is subject to a uniform distribution. The average value of vertex curvatures is computed at first, and the vertices with curvatures that are far from the average value are sampled with higher probability (i.e., the vertices with more special surface patterns are more likely to be sampled).

In the experiments, we randomly split the 1, 000 instances of each object into train/val/test splits of 800/190/10, respectively. Similarly, in the real experiments, we choose 53 objects from ObjectFolder Real and randomly split the instances of each object by 8:1:1.

#### Results on ObjectFolder

<table>
    <tr>
        <td>Method</td>
        <td>Vision</td>
        <td>Touch</td>
        <td>Audio</td>
        <td>V+T</td>
        <td>V+A</td>
        <td>T+A</td>
        <td>V+T+A</td>
    </tr>
    <tr>
        <td>RANDOM</td>
        <td>47.32</td>
        <td>47.32</td>
        <td>47.32</td>
        <td>47.32</td>
        <td>47.32</td>
        <td>47.32</td>
        <td>47.32</td>
    </tr>
  <tr>
        <td>Point Filtering</td>
        <td>-</td>
        <td>4.21</td>
        <td>1.45</td>
        <td>-</td>
        <td>-</td>
        <td>3.73</td>
        <td>-</td>
    </tr>
  <tr>
        <td>MCR</td>
        <td>5.03</td>
        <td>23.59</td>
        <td>4.85</td>
        <td>4.84</td>
        <td>1.76</td>
        <td>3.89</td>
        <td>1.84</td>
    </tr>
</table>

#### Results on ObjectFolder Real

<table>
    <tr>
        <td>Method</td>
        <td>Vision</td>
        <td>Touch</td>
        <td>Audio</td>
        <td>Fusion</td>
    </tr>
    <tr>
        <td>RANDOM</td>
        <td>50.57</td>
        <td>50.57</td>
        <td>50.57</td>
        <td>50.57</td>
    </tr>
  	<tr>
        <td>MCR</td>
        <td>12.30</td>
        <td>32.03</td>
        <td>35.62</td>
        <td>12.00</td>
    </tr>
</table>

