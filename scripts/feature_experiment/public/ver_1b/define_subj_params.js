// define paramters for the subject... shuffle these
var choice_images = ["Stimuli/Evan_Stimuli/fractal_A.png",
                  "Stimuli/Evan_Stimuli/fractal_B.png",
                  "Stimuli/Evan_Stimuli/fractal_C.png",
                ];
var practice_image = "Stimuli/Evan_Stimuli/fractal_D.png";

var safe_first = true;

if (safe_first){
  var feature_probs = feature_prob_safe_first;
  var rewards = rewards_safe_first;
}else{
  var feature_probs = feature_prob_danger_first;
  var rewards = rewards_danger_first;
}

jsPsych.data.addProperties({safe_first: safe_first});
