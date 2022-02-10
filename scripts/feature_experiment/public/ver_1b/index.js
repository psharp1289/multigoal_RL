
// firebase stuff
firebase.firestore().enablePersistence().catch(function(err) {
    if (err.code == 'failed-precondition') {
        // Multiple tabs open, persistence can only be enabled
        // in one tab at a a time.
    } else if (err.code == 'unimplemented') {
        // The current browser does not support all of the
        // features required to enable persistence
    }
});

firebase.auth().signInAnonymously();

var uid;


// Consent form
var check_consent = function (elem) {
  if ($('#consent_checkbox1').is(':checked') && $('#consent_checkbox2').is(':checked') &&
      $('#consent_checkbox3').is(':checked') && $('#consent_checkbox4').is(':checked') &&
      $('#consent_checkbox5').is(':checked') && $('#consent_checkbox6').is(':checked') &&
      $('#consent_checkbox7').is(':checked'))
      {
          // When signed in, get the user ID
          firebase.auth().onAuthStateChanged(function(user) {
            if (user) {
              uid = user.uid;
              task(uid);
            }
          });
      }

  else {
      alert("Unfortunately you will not be unable to participate in this research study if you do " +
          "not consent to the above. Thank you for your time.");
      return false;
  }
};

function getQueryVariable(variable)
{
       var query = window.location.search.substring(1);
       var vars = query.split("&");
       for (var i=0;i<vars.length;i++) {
               var pair = vars[i].split("=");
               if(pair[0] == variable){return pair[1];}
       }
       return(false);
}


function task(uid){

    if (window.location.search.indexOf('PROLIFIC_PID') > -1) {
        var subjectID = getQueryVariable('PROLIFIC_PID');
    }
    else {
        var subjectID = Math.floor(Math.random() * (2000000 - 0 + 1)) + 0; // if no prolific ID, generate random ID (for testing)
    }

    var db = firebase.firestore();
    var run_name = 'run1b';

    // record new date and start time
    db.collection('featuretask').doc(run_name).collection('subjects').doc(uid).set({
        subjectID: subjectID,
        safe_first: safe_first
    })

    // record new date and start time
    db.collection('featuretask').doc(run_name).collection('subjects').doc(uid).collection('taskdata').doc('start').set({
        subjectID: subjectID,  // this refers to the subject's ID from prolific/
        date: new Date().toLocaleDateString(),
        start_time: new Date().toLocaleTimeString(),
        safe_first: safe_first
    })


    // this script constructs a 'timeline' - an array of dictionaries (the things in brackets, {})...
    timeline = [];

    // this is a jspsych defined plug-in which
    var full_screen = { // this plugin will prompt the full screen
        type: 'fullscreen',
        fullscreen_mode: true
    };

    // place choie stim, wait for response
    n_choice_trials = Object.keys(rewards).length; // define

    var first_third_break = {
        type: 'html-button-response',
        timing_post_trial: 0,
        choices: ['Continue'],
        is_html: true,
        stimulus: "Great work. You are a third of the way through the task.",
    }

    var second_third_break = {
        type: 'html-button-response',
        timing_post_trial: 0,
        choices: ['Continue'],
        is_html: true,
        stimulus: "Great work. You are two thirds of the way through the task.",
    }

    timeline = [full_screen];
    timeline = timeline.concat(intro_w_trials);

    for (var i = 0; i < n_choice_trials; i++){
        var choice_trial = { // this calls the plugin that i made in - jspsych-evan-explugin.js
          // it sets parameters for the plugin
          type: 'evan-feature33',
          feature_rewards: rewards[i],
          c1_image: choice_images[0],
          c2_image: choice_images[1],
          c3_image: choice_images[2],
          c1_feature_probs: feature_probs[i]["s_0"],
          c2_feature_probs: feature_probs[i]["s_1"],
          c3_feature_probs: feature_probs[i]["s_2"],
          choice_prompt: true,
          single_choice_option: false
        }
        timeline.push(choice_trial)

        if (i == Math.round(n_choice_trials/3)){
            timeline.push(first_third_break);
        }
        if (i == Math.round(2*n_choice_trials/3)){
            timeline.push(second_third_break);
        }
    }


    // notice anything
    var notice_question = {
      type: 'survey-text',
      questions: [
        {prompt: "Did you notice anything differently about the first and second halves of the block ?"}
      ],
      data:{trial_num: 'Q', Q_name: 'notice'}
    };
    timeline.push(notice_question);

    // could you track
    var track_question = {
      type: 'evan-quiz',
      questions: [
        {prompt: "Did you feel like you could track where the all the aliens were?", options: ['yes', 'no']}
      ],
      data:{trial_num: 'Q', Q_name: 'track1'}
    };
    timeline.push(track_question);

    // could you track2
    var track_question2 = {
      type: 'evan-quiz',
      questions: [
        {prompt: "If not, did you find yourself just focusing on one or two of the aliens?", options: ['yes', 'no', 'N/A']}
      ],
      data:{trial_num: 'Q', Q_name: 'track2'}
    };
    timeline.push(track_question2);

    // notice anything
    var strat_question = {
      type: 'survey-text',
      questions: [
        {prompt: "Can you describe the strategy you used to do well on the task?"}
      ],
      data:{trial_num: 'Q', Q_name: 'strat'}
    };
    timeline.push(strat_question);

    // compute bonus for the main task...
    var end_screen = {
        type: 'html-button-response',
        timing_post_trial: 0,
        choices: ['End Task'],
        is_html: true,
        stimulus: function(){
            var task_data = jsPsych.data.get().json()

          //var random_total_points = jsPsych.randomization.sampleWithoutReplacement(total_points_arr, 1);

          var string = 'You have finished the task. Thank you for your contribution to science! \
                <b> PLEASE CLICK END TASK TO SUBMIT THE TASK TO PROLIFIC </b>.';

          db.collection('featuretask').doc(run_name).collection('subjects').doc(uid).collection('taskdata').doc('data').set({
            //bonus_points: random_total_points,
            data:  task_data
          })
       return string;
      },
       on_finish: function(){
         window.location =  "https://app.prolific.co/submissions/complete?cc=E3FCD9EE"
       }
    }

    timeline.push(end_screen)

    var instruc_images = instruction_pagelinks_a.concat(instruction_pagelinks_b);
    instruc_images = instruc_images.concat(instruction_pagelinks_c);
    instruc_images = instruc_images.concat(choice_images)
    instruc_images = instruc_images.concat(practice_image)

    jsPsych.init({ // this runs the exmperiment and does a local save of the results.
      timeline: timeline,
      preload_images: instruc_images,
      show_preload_progress_bar: false,
      //on_finish: function() {
        //jsPsych.data.get().localSave('csv','results.csv');
    //}
    });
}


document.getElementById('header_title').innerHTML = "Online studies in learning, decision-making and cognition: Information and consent";
document.getElementById('consent').innerHTML = "        <p><b>Who is conducting this research study?</b><p>\n" +
    "        <p>\n" +

    "        This research is being conducted by the Division of Psychiatry and the Max Planck UCL Centre for Computational Psychiatry\n" +
    "        and Ageing Research at University College London, London, UK. The lead researchers for this project is\n" +
    "        <a href=\"mailto:p.sharp@ucl.ac.uk\">Dr Paul Sharp</a>. This study has been approved by the UCL Research Ethics Committee\n" +
    "        (project ID number 16639/001) and is funded by the Max Planck Society.\n" +
    "        </p>\n" +
    "\n" +
    "        <p><b>What is the purpose of this study?</b><p>\n" +
    "        <p>\n" +
    "        We are interested in how the adult brain controls learning and decision-making. This research aims to provide\n" +
    "        insights into how the healthy brain works to help us understand the causes of a number of different medical\n" +
    "        conditions.\n" +
    "        </p>\n" +
    "\n" +
    "        <p><b>Who can participate in the study?</b><p>\n" +
    "        <p>\n" +
    "            You must be 18 or over to participate in this study. Please confirm this to proceed.\n" +
    "        </p>\n" +
    "            <label class=\"container\">I confirm I am over 18 years old\n" +
    "                <input type=\"checkbox\" id=\"consent_checkbox1\">\n" +
    "                <span class=\"checkmark\"></span>\n" +
    "            </label>\n" +
    "        <br>\n" +
    "\n" +
    "        <p><b>What will happen to me if I take part?</b><p>\n" +
    "        <p>\n" +
    "            You will play one or more online computer games, which will last approximately 45 minutes. You will receive\n" +
    "            at least 6 GBP for helping us out with an opportunity for an additional bonus depending on your choices. The amount may vary with the decisions you make in the games.\n" +
    "            Remember, you are free to withdraw at any time without giving a reason.\n" +
    "        </p>\n" +
    "\n" +
    "        <p><b>What are the possible disadvantages and risks of taking part?</b><p>\n" +
    "        <p>\n" +
    "            The task will you complete does not pose any known risks.\n" +
    "        </p>\n" +
    "\n" +
    "        <p><b>What are the possible benefits of taking part?</b><p>\n" +
    "        <p>\n" +
    "            While there are no immediate benefits to taking part, your participation in this research will help us\n" +
    "        understand how people make decisions and this could have benefits for our understanding of mental health problems.\n" +
    "        </p>\n" +
    "\n" +
    "        <p><b>Complaints</b><p>\n" +
    "        <p>\n" +
    "        If you wish to complain or have any concerns about any aspect of the way you have been approached or treated\n" +
    "        by members of staff, then the research UCL complaints mechanisms are available to you. In the first instance,\n" +
    "        please talk to the <a href=\"mailto:e.russek@ucl.ac.uk\">researcher</a> or the chief investigator\n" +
    "        (<a href=\"mailto:q.huys@ucl.ac.uk\">Dr Quentin Huys</a>) about your\n" +
    "        complaint. If you feel that the complaint has not been resolved satisfactorily, please contact the chair of\n" +
    "        the <a href=\"mailto:ethics@ucl.ac.uk\">UCL Research Ethics Committee</a>.\n" +
    "\n" +
    "        If you are concerned about how your personal data are being processed please contact the data controller\n" +
    "        who is <a href=\"mailto:data-protection@ucl.ac.uk\">UCL</a>.\n" +
    "        If you remain unsatisfied, you may wish to contact the Information Commissioner Office (ICO).\n" +
    "        Contact details, and details of data subject rights, are available on the\n" +
    "        <a href=\"https://ico.org.uk/for-organisations/data-protection-reform/overview-of-the-gdpr/individuals-rights\">ICO website</a>.\n" +
    "        </p>\n" +
    "\n" +
    "        <p><b>What about my data?</b><p>\n" +
    "        <p>\n" +
    "        This local privacy notice sets out the information that applies to this particular study. Further information on how UCL uses participant information can be found in our general privacy notice: \n \n " +
    "        For participants in research studies, click <a href=\"https://www.ucl.ac.uk/legal-services/privacy/ucl-general-research-participant-privacy-notice\">here</a>    \n \n   " +
    "        The information that is required to be provided to participants under data protection legislation (GDPR and DPA 2018) is provided across both the local and general privacy notices. \n" +

    "        To help future research and make the best use of the research data you have given us (such as answers" +
    "        to questionnaires) we may keep your research data indefinitely and share these.  The data we collect will\n" +
    "        be shared and held as follows:<br> \n" +
    "        In publications, your data will be anonymised, so you cannot be identified. <br> \n" +
    "        In public databases, your data will be anonymised (your personal details will be removed and a code used e.g. 00001232, instead of your User ID) <br>" +
    "\n" +
    "         Personal data is any information that could be used to identify you, such as your User ID.  When we collect your data, your User ID will be replaced with a non-identifiable random ID number. No personally identifying data will be stored \n" +
    "        If there are any queries or concerns please do not hesitate to contact <a href=\"mailto:p.sharp@ucl.ac.uk\">Dr Paul Sharp</a>.\n" +
    "        </p>\n" +
    "\n" +
    "        <p><b>If you are happy to proceed please read the statement below and click the boxes to show that you\n" +
    "            consent to this study proceeding</b><p>\n" +
    "\n" +
    "        <label class=\"container\">I have read the information above, and understand what the study involves.\n" +
    "            <input type=\"checkbox\" id=\"consent_checkbox2\">\n" +
    "            <span class=\"checkmark\"></span>\n" +
    "        </label>\n" +
    "\n" +
    "        <label class=\"container\">I understand that my anonymised/pseudonymised personal data can be shared with others\n" +
    "            for future research, shared in public databases and in scientific reports.\n" +
    "            <input type=\"checkbox\" id=\"consent_checkbox3\">\n" +
    "            <span class=\"checkmark\"></span>\n" +
    "        </label>\n" +
    "\n" +
    "        <label class=\"container\">I understand that I am free to withdraw from this study at any time without\n" +
    "            giving a reason and this will not affect my future medical care or legal rights.\n" +
    "            <input type=\"checkbox\" id=\"consent_checkbox4\">\n" +
    "            <span class=\"checkmark\"></span>\n" +
    "        </label>\n" +
    "\n" +
    "        <label class=\"container\">I understand the potential benefits and risks of participating, the support available\n" +
    "            to me should I become distressed during the research, and who to contact if I wish to lodge a complaint.\n" +
    "            <input type=\"checkbox\" id=\"consent_checkbox5\">\n" +
    "            <span class=\"checkmark\"></span>\n" +
    "        </label>\n" +
    "\n" +
    "        <label class=\"container\">I understand the inclusion and exclusion criteria in the Information Sheet.\n" +
    "            I confirm that I do not fall under the exclusion criteria.\n" +
    "            <input type=\"checkbox\" id=\"consent_checkbox6\">\n" +
    "            <span class=\"checkmark\"></span>\n" +
    "        </label>\n" +
    "\n" +
    "        <label class=\"container\">I agree that the research project named above has been explained to me to my\n" +
    "            satisfaction and I agree to take part in this study\n" +
    "            <input type=\"checkbox\" id=\"consent_checkbox7\">\n" +
    "            <span class=\"checkmark\"></span>\n" +
    "        </label>\n" +
    "\n" +
    "        <br><br>\n" +
    "        <button type=\"button\" id=\"start\" class=\"submit_button\">continue</button>\n" +
    "        <br><br>";


document.getElementById("start").onclick = check_consent;

if (/Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent)) {
    alert("Sorry, this experiment does not work on mobile devices");
    document.getElementById('consent').innerHTML = "";
}
