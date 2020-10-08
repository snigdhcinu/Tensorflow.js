// // STEP 1 :- TRAINING THE MODEL

	// Telling that the neural network that we are creating is sequential
	const model=tf.sequential();

	/* Adding a single dense layer to our sequence,
	 	units = 1 implies only one neuron is there in our dense layer

	*/
	model.add(tf.layers.dense({units:1,inputShape:[1]}));

	/* We compile our model
		loss = meanSquaredError implies that we are using MSE as our loss function, 
			   which is good for linear relationships like this one.
		sgd = implies stocastic gradient descent
	*/
	model.compile({loss:'meanSquaredError',optimizer:'sgd'});

	// outputting the summary of the model
	model.summary();


	// adding 2d array, 1st is the data, and second is the dimension of the input array.
	// i.e., in tensor2d we have the data, and the shape of the data.
	const xs = tf.tensor2d([-1.0,0.0,1.0,2.0,3.0,4.0],[6,1])
	const ys = tf.tensor2d([-3.0,-1.0,2.0,3.0,5.0,7.0],[6,1])


// STEP 2 :- TRAINING THE MODEL

	/*
		Training should be done in async function, since it will take indeterminant
		amount of time, and we don't want to wait for it to compile, to load the DOM.	

		doTraining() is the async function, which when trained will alert the
		prediction.
	*/

	doTraining(model).then(() => {
		alert(model.predict(tf.tensor2d([10],[1,1])));
	})

	// Defining the async function
	async function doTraining(model){
		const history = 
			await model.fit(xs,ys,
			{
				epochs:500,
				callbacks:{
					// async anonymous function attached as an object key value.
					// callback attached which happens/executed at the end of every epoch
					onEpochEnd: async(epochs,logs) => {
						console.log(`Epoch: ${epochs} Loss: ${logs.loss}`);
					}
				}
			})
	}