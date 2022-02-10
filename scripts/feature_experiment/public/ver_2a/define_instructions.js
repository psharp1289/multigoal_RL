// define the instruction pages... note whether these are JPG or jpg
var instruction_pagelinks_a = ['Stimuli/Feature_Task_Instructions/Slide1.jpg',
                            'Stimuli/Feature_Task_Instructions/Slide2.jpg',
                            'Stimuli/Feature_Task_Instructions/Slide3.jpg'];

/// then comes the practice trials -- partial
var instruction_pagelinks_b = ['Stimuli/Feature_Task_Instructions/Slide5.jpg',
                            'Stimuli/Feature_Task_Instructions/Slide6.jpg',
                            'Stimuli/Feature_Task_Instructions/Slide7.jpg',
                            'Stimuli/Feature_Task_Instructions/Slide8.jpg'];

var instruction_pagelinks_c = ['Stimuli/Feature_Task_Instructions/Slide9.jpg'];

var pages_a = [];
for (var i = 0; i < instruction_pagelinks_a.length; i++){
    pages_a.push('<img src= "'+ instruction_pagelinks_a[i] +  '" alt = "" >')
}
var pages_b = [];
for (var i = 0; i < instruction_pagelinks_b.length; i++){
    pages_b.push('<img src= "'+ instruction_pagelinks_b[i] +  '" alt = "" >')
}
var pages_c = [];
for (var i = 0; i < instruction_pagelinks_c.length; i++){
    pages_c.push('<img src= "'+ instruction_pagelinks_c[i] +  '" alt = "" >')
}

var instruction_pages_a = {
    type: 'instructions',
    pages: pages_a,
    show_clickable_nav: true
}

var instruction_pages_b = {
    type: 'instructions',
    pages: pages_b,
    show_clickable_nav: true
}

var instruction_pages_c = {
    type: 'instructions',
    pages: pages_c,
    show_clickable_nav: true
}

var n_practice_trials_1 = 5;
var practice_trials_1 = [];
for (var i = 0; i < n_practice_trials_1; i++){
  var this_trial = {
    type: 'evan-feature22',
    feature_rewards: [-3,3], // this won't be shown...
    c1_image: choice_images[0],
    c2_image: choice_images[1],
    c1_feature_probs: [1,0],
    c2_feature_probs: [.2,.7],  ///
    choice_prompt: true,
    single_choice_option: true
  }
  practice_trials_1.push(this_trial);
}


//// ...
var practice_feature_rewards = [[-3, 0], [0, 3], [0,3], [-3, 0], [-3, 0], [0,3]];
var n_practice_trials_2 = 5;//practice_feature_rewards.length;
var practice_trials_2 = [];
for (var i = 0; i < n_practice_trials_2; i++){
  var this_trial = {
    type: 'evan-feature22',
    feature_rewards:  practice_feature_rewards[i], // this won't be shown...
    c1_image: choice_images[0],
    c2_image: choice_images[1],
    c1_feature_probs: [.6,.3], /// adjust these?
    c2_feature_probs: [.3,.5],
    choice_prompt: true,
    single_choice_option: false,
		update_prompt: true
  }
  practice_trials_2.push(this_trial);
}

/// BUILD THE QUIZ...

var questions = ["How many types of aliens are in the task?", // 1
				"What determines your bonus?", // 2
				"Do the chances that certain aliens are behind certain doors change over the course of the task?", // 3
				"What does the gold number displayed on the first decision screen indicate?", // 4
				"What determines how many minerals you receive after opening a door?"];

var options1 =  ['1 ', '2', '3', '4', '5'];
var correct1 = 1;

var options2 =  ['The total number of minerals gained over the course of the entire task.',
								'The number of minerals gained minus the number of minerals lost on 5 randomly selected decisions.',
								'Number of times the silver alien was encountered.',
								'It is random.'];
var correct2 = 1;

var options3 =  ['Yes.',
								'No.'];
var correct3 = 0;

var options4 =  ['The number of minerals that the gold alien will provide (if positive) or take away (if negative)',
								'The number of minerals that the black alien will provide (if positive) or take away (if negative)',
								'The chances that the black alien is behind the left door'];
var correct4 = 0;

// build the quiz...
var corr_string = '{"Q0":' + '"'+options1[correct1]+'",' + '"Q1":' + '"'+options2[correct2]+'",'
    + '"Q2":' + '"'+options3[correct3]+'",' + '"Q3":' + '"'+options4[correct4]+'"'+'}';

var preamble = ["<p align='center'><b>Please answer every question. Answering 'I do not know' or answering incorrectly will require you return to the beginning of the instructions. </b></p>"];

var instruction_correct = false;
var instruction_check = {
	type: "evan-quiz",
      preamble: preamble,
      questions: [
          {prompt: "<b>Question 1</b>: " + questions[0],
                  options: options1, required: true},
          {prompt: "<b>Question 2</b>: " + questions[1],
                      options: options2, required: true},
          {prompt: "<b>Question 3</b>: " + questions[2],
                      options: options3, required: true},
          {prompt: "<b>Question 4</b>: " + questions[3],
                          options: options4, required: true}
  		],
  		on_finish: function(data) {
					console.log(data.responses)
					console.log(corr_string)
	      if( data.responses == corr_string){
	          action = false;
	          instruction_correct = true;
	      }else{
			var post_choices = data.choice_idxs
			// this is global
			incor_questions = ['<br> </br'];
			var correct_choices = [correct1, correct2, correct3, correct4];
			for (var i = 0; i < correct_choices.length; i++){
				if (correct_choices[i] != post_choices[i]){
					incor_questions.push('<br>' + questions[i] + '</br>');
				}
			}
			data.incor_questions = incor_questions;
		}
	}
}

/* define a page for the incorrect response */
var showsplash = true;
var splash_screen = {
	type: 'html-button-response',
    timing_post_trial: 0,
	//    button_html: '<button class="jspsych-btn" style="display:none">%choice%</button>',
    choices: ['Click here to read the instructions again'],
    is_html: true,
    stimulus: function(){
			var incor_q = jsPsych.data.get().last(1).select('incor_questions').values
			var next_stimulus = 'The following questions were answered incorrectly: ' + incor_q;
			return next_stimulus
		}
}

/* ...but push it to a conditional node that only shows it if the response was wrong */
var conditional_splash = {
  timeline: [splash_screen],
  conditional_function: function(data) {
	return !instruction_correct // skip if correct
	}
}


	//////
	var intro_w_trials = [];
	intro_w_trials.push(instruction_pages_a);
	intro_w_trials = intro_w_trials.concat(practice_trials_1);
	intro_w_trials.push(instruction_pages_b);
	intro_w_trials = intro_w_trials.concat(practice_trials_2);
	intro_w_trials.push(instruction_pages_c);
	intro_w_trials.push(instruction_check);
	intro_w_trials.push(conditional_splash);

	var intro_loop = [];
	intro_loop.push(instruction_pages_a);
	intro_loop.push(instruction_pages_b);
	intro_loop.push(instruction_pages_c);
	intro_loop.push(instruction_check);
	intro_loop.push(conditional_splash);

	/* finally, add the entirety of this introductory section to a loop node ... */
	var loop_node = {
	  timeline: intro_loop,
	  conditional_function: function(data) {
	  	return !instruction_correct // skip if correct
	},
	  loop_function: function(data) {
		var action = true;
		return !instruction_correct // stop looping if correct
		}
	}

	intro_w_trials.push(loop_node);

	var finish_instruc_screen = {
		type: 'html-button-response',
		timing_post_trial: 0,
		//    button_html: '<button class="jspsych-btn" style="display:none">%choice%</button>',
		choices: ['Begin the task!'],
		is_html: true,
		stimulus: 'You passed the quiz! Great work. The task will take about 30 minutes. Press the button to begin.'
	}

	intro_w_trials.push(finish_instruc_screen);
