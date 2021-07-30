# Syllabus  

## Course overview and goals

This course will introduce students with basic Python and Statistics experience to standard data science methods and procedures.

### Goals

At the end of the course you should have a working understanding of how to work with dataset in Python, and the basic goals, implementation, and usage of standard data science approaches.

- Read in, clean, wrangle, process, and tidy data of different types   
- Understand how model fitting and prediction works  
- Evaluate classification, regression   
- Know when to cluster, reduce dimensionality and how

### Strategy

This course is structured to help you get there.  The basic premise we start from is that programming is a skill, and to acquire a skill you need lots and lots of practice (see [expectations](expectations.md)).  In lecture, we will introduce new basic elements, and will try them together as a class.  In labs, we will work in groups to combine these elements into small programs with instructor help.  In problem sets, you will apply these skills on your own to solve specific problems.  As much as possible, we will use interesting Computational Social Science problems as motivating problems, but we are constrained first and foremost by finding problems appropriate for your developing Python skills.

## Course Information

**Summer Session 2, 2021**  

### Instructors

| Role     | Name                | email    | Office hours  |
| ----------------: | :---------| -----------| ------------ |   
| **Instructor** | Ed Vul  | [evul@ucsd.edu](mailto:evul@ucsd.edu) | [Fridays 3:00-4:30PM](https://ucsd.zoom.us/my/edvul) |   

### All the links

- Main {{ url_website }}: contains all the materials and links    
- Lecture / Lab zoom: {{ url_lecture }}: M/W 11am - 3pm PST (conducted live, recordings posted on canvas.)   
- {{ url_canvas }}: used to post grades and recordings of lectures/labs   
- {{ url_campuswire }}: **CODE: 2003** used for all communication: discussion, Q&A, announcements etc.  Email the instructor if you cannot access.    
- {{ url_datahub }}: used to submit comleted labs and problem sets  

### Lectures and labs

Lectures and labs take place at their scheduled time (with some intermixing): Mondays and Wednesdays (11am - 3pm) at this zoom link:  {{ url_lecture }}.  Instructor-led lecture, and student-led lab practice are intermixed.

Recordings of these will be made available on  {{ url_canvas }}      

### Materials

- All materials will be provided via this website and the links above.
  
- No textbook is required atop the lectures notes here, but we provide recommendations for some paid and free [extracurricular resources](resources.md)     
  
- No local software is required (as we will use remotely hosted jupyter notebooks).  If you want to install a local copy, we recommend the bundled [anaconda distribution](https://www.anaconda.com/products/individual) of Python 3  

# Schedule

This class is divided into 10 parts, corresponding to the normal 10 weeks of the academic year. Each part has associated lectures and labs.

**During summer session we cover two parts per week (one per day).**  
  
- each week has a problem set motivating (and testing) the material that we cover that week.  

- each day contains lecture (introducing the concepts and methods), and lab (hands on, group practice culminating in a submitted notebook graded for completion)  

| Date | Topics | Assignment due |
| ---- | ------ | -------------- |
| 2021-08-02 | [Review: Python, notebooks](../lectures/P01) | |
| 2021-08-03 |    | Lab 1 due |
| 2021-08-04 | [Data: Numpy and Pandas](../lectures/P02) | |
| 2021-08-05 |    | Lab 2 due |
| 2021-08-08 |    | pset 1 due |
| 2021-08-09 | [Plots: matplotlib](../lectures/P03) | |
| 2021-08-10 |    | Lab 3 due |
| 2021-08-11 | [Getting and managing data](../lectures/P04) | |
| 2021-08-12 |    | Lab 4 due |
| 2021-08-15 |    | pset 2 due |
| 2021-08-16 | [Models 1: form, parameters, loss, fitting](../lectures/P05) | |
| 2021-08-17 |    | Lab 5 due |
| 2021-08-18 | [Models 2: evaluation, cross-validation](../lectures/P06) | |
| 2021-08-19 |    | Lab 6 due |
| 2021-08-22 |    | pset 3 due |
| 2021-08-23 | [Classification](../lectures/P07) | |
| 2021-08-24 |    | Lab 7 due |
| 2021-08-25 | [Regression](../lectures/P08) | |
| 2021-08-26 |    | Lab 8 due |
| 2021-08-29 |    | pset 4 due |
| 2021-08-30 | [Clustering](../lectures/P09) | |
| 2021-08-31 |    | Lab 9 due |
| 2021-09-01 | [Dimensionality reduction](../lectures/P10) | |
| **2021-09-04** |    | final pset due |


## Grading

### Basis

You are evaluated based on:   
- 35% 4 weekly problem sets (weeks 1-4)   
- 15% final (week 5)   
- 40% 10 labs  (2 per week)
- 10% pro-social behavior 

**Labs:** Labs are short, simple exercises designed to be completed during the scheduled lab time, with interactive help from instructors and other students.  Labs are completed by turning them in on datahub.  Labs are due by the end of the day following lab.  This window is wide so that people who cannot attend lab, or otherwise do not complete the work during lab, can submit on their own schedule.  That said, *it is very much advised that you attend lab to complete the activities and get interactive help!*

**Problem Sets:** Are longer, weekly assignments.  They are due by the end of the end of the day on Sunday of a given week.  You are to complete each problem set **on your own**.  You are advised to *start early*

**Final:** The final is a more involved, more integrated problem set, due at the end of week 5.

**Pro-Social Behavior:** This component of your grade is based on doing things that help the instructors and other students, and generally creating a positive class environment.  This includes things like: showing up and participating during lectures and labs, participating in campuswire discussion (asking good questions, answering others' questions), demonstrating an interest in learning, not just maximizing your grade, etc. 

### Letter grades

Letter grades will be based on the percentage of total points earned across the items above.  Having encoded the percentage in the variable `percent`, we can obtain the grade as follows:  

```python
if percent >= 90: 
    letter = 'A'   
    remainder = percent - 90
    
if 90 > percent >= 80:
    letter = 'B'
    remainder = percent - 80

if 80 > percent >= 70:
    letter = 'C'
    remainder = percent - 70

if 70 > percent >= 60:
    letter = 'D'
    remainder = percent - 60

if 60 > percent:
    letter = 'F'
    remainder = 5

if remainder >= 7:
    modifier = '+'
elif remainder < 3:
    modifier = '-'
else:
    modifier = ''

grade = letter + modifier
```

### Assignment scores

Assignment scores will be made available via assignment feedback on datahub (where they are submitted)

### Manual Regrades

Problem sets and labs are scored using [nbgrader](https://nbgrader.readthedocs.io/en/stable/) on [UCSD datahub](https://datahub.ucsd.edu/hub/login).  Some parts are graded automatically by computer, and some parts are graded manually by a human.

We will work hard to grade everyone fairly and return assignments quickly. And, we know you also work hard and want you to receive the grade you’ve earned. Occasionally, grading mistakes do happen, and it’s important to us to correct them. If you think there is a mistake in your grade on an assignment, post privately on Campuswire to “Instructors & TAs” using the “regrades” tag within 72 hours. This post should include evidence of why you think your answer was correct and should point to the specific part of the assignment in question.

Note that points will not be rewarded if you fail to follow instructions. For example, if the instructions say to name the variable `orange` and you name it `ornage` (misspelled), you will not be rewarded credit upon regrade. This is because (1) following instructions and being detail-oriented in general, (2) referring to things by their correct names, and getting other minor technicalitieis right is *essential* to programming.

## Questions, feedback, and communication

The instructors can be reached in the following ways:   

- Drop in during scheduled **office hours** (see [syllabus](syllabus.md) for links and schedule). 

- Public message on {{ url_campuswire }}.   

- Private "Instructors & TAs" message on {{ url_campuswire }}  

- Direct message to specific instructor on {{ url_campuswire }}  

Outside of office hours, all communication should happen over {{ url-campuswire }}.  Email is reserved for the unanticipated circumstances when campuswire is down, or you cannot access it.  In that case, [email the instructor](mailto:evul@ucsd.edu) about in inability to access Campuswire.

### Specific types of questions / comments

- **questions about course logistics:** First, check the syllabus and the detailed how-to pages on the {{ url_website }}. If you can't find the answer there, first ask a classmate. If still unsure, post on Campuswire in the General tag.
  
- **questions about course content:** these are awesome! We want everyone to see them, be able to answer them, and have their questions answered too, so post these to Campuswire with an appropriate tag!  

- **my code produces an error that I cannot fix:** follow the [debugging instructions](debugging.md) to find a minimal reproducible example and fill out the debugging question checklist, then post on Campuswire in the "Python" category or the relevant "Problem Set" category.

- **assignment clarification question:** Ask in the appropriate "Problem Set" or "Labs" category.  
  
- **a technical assignment question:** Come to office hours (or post to Campuswire). Answering technical questions is often best accomplished 'in person' where we can discuss the question and talk through ideas. However, if that is not possible, post your question to Campuswire. Be as specific as you can in the question you ask. And, for those answering, help your classmates as much as you can without just giving the answer. Help guide them, point them in a direction, provide pseudo code, but do not provide code that answers assignment questions.  
  
- **been stuck on something for a while (>30min) and aren't even really sure where to start** - Programming can be frustrating and it may not always be obvious what is going wrong or why something isn't working. That's OK - we've all been there! IF you are stuck, you can and should reach out for help, even if you aren't exactly sure what your specific question is. To determine when to reach out, consider the 2-hour rule. This rule states that if you are stuck, work on that problem for an hour. Then, take a 30 minute break and do something else. When you come back after your break, try for another 30 minutes or so to solve your problem while working through our [debugging](debugging.md) checklist. If you are still completely stuck, stop and contact us (office hours, post on Campuswire). If you don't have a specific question, include the information you have (what you're stuck on, the [debugging checklist](debugging.md)).
  
- **questions about a grade** - Post on Campuswire with the "Regrades" tag in a private post to "Instructors & TAs".
  
- **something super cool to share related to class or want to talk about a topic in further depth** - come to office hours, post in General, or send in a DM to the instructors!
  
### Campuswire Rules

Campuswire is an incredible resource for technical classes. It gives you a place to post questions and an opportunity to answer others' questions. We do our very best as an instructional staff to make sure each and every question is answered in a timely manner. We also want to make sure this platform is being used to learn and not thwarting anyone's education. To make all of this possible, there are a few rules for this course's campuswire:  

1. Before posting your question, look at questions that have already been posted to avoid duplicates.   
2. If posting about an assignment, note title should have assignment number, question number, and 1-2 words about the question. (i.e. PS01 Q1 Variable Naming)    
3. Never post an answer to or code for an assignment on a public post. Pseudocode is encouraged for public posts. If you must include code for an assignment, make this post private (to "Instructors & TAs" only) on Campuswire.   
   
4. Your post must include not only your question/where you're stuck, but also what you've already done to try to solve it so far and what resources (class notes, online URLs, etc.) you used to try to answer the question up to this point.  See how to ask [debugging questions](debugging.md).



## Remote learning

For some of you, remote learning is new. For others, you've got a bit of practice. For all of us, there is a lot going on in the world. While students have always been under a fair amount of pressure and stress, the struggles students may encounter this quarter (for a whole bunch of different reasons) may go beyond what is typical. I want you all to know that I fully understand this and am here to help you succeed. 

While regular deadlines have been established to help keep you all on track, I want you to know up front that I am a very reasonable person. While I ask that you all do your best to meet deadlines that have been set, know that if you're struggling, I absolutely want you to reach out to let me know, to ask for an extension, or to discuss some other accommodation.

Please take care of yourselves and one another, and I'll work as hard as needed to ensure success for all students this quarter.

### Remote technology

If you do not have consistent access to the technology needed to fully access remote instruction options, please use the form below to request a loaner laptop for the period during which you will be learning remotely due to the COVID-19 pandemic: [https://eforms.ucsd.edu/view.php?id=490887](https://eforms.ucsd.edu/view.php?id=490887). (For any issues that you may have, please email [vcsa@ucsd.edu](mailto:vcsa@ucsd.edu) and they will work to assist you.)


### Remote Lectures and Labs

Attendance will be neither required nor incentivized for any part of the course this quarter. This policy is in place because we do not want to disadvantage students working in different time zones. While lectures and coding labs will take place during their scheduled times, there *will* be options for students to complete all work asynchronously.

Lectures will take place at their scheduled time for those who are able to attend. As typically occurs in CSS 1, students will be encouraged to follow along with the notes, will be given time to complete small coding challenges during lecture on their own, and will have the opportunity to see their classmates thoughts during lecture.

However, every lecture will also be recorded and shared so that students who are not able to or choose not to watch during the scheduled class time are still able to receive and digest all class materials. If a lecture recording ever fails during class, instructors will re-record a lecture, ensuring all students have access to the material. Lecture and Lab recordings will be available on {{ url_canvas }} in the Media Gallery.




## UCSD policies & resources

### Academic Integrity

[From UCSD Academic Integrity office](https://academicintegrity.ucsd.edu/take-action/promote-integrity/faculty/syllabus-statements.html#General-statement-on-academic-i)

> Integrity of scholarship is essential for an academic community. The University expects that both faculty and students will honor this principle and in so doing protect the validity of University intellectual work. For students, this means that all academic work will be done by the individual to whom it is assigned, without unauthorized aid of any kind.

[Please read the full UCSD policy](http://senate.ucsd.edu/Operating-Procedures/Senate-Manual/Appendices/2)

You are encouraged to work together and help one another for *labs*. However, you are personally responsible for the work you submit. It is your responsibility to ensure you understand everything you've submitted.

**You must work independently on the problem sets and the final.**  You may ask and answer [debugging questions](debugging.md) on campuswire, but doing work for another student or providing assistance outside of public questions on campuswire on the problem sets or final project will be treated as a violation of academic integrity and you will be referred for disciplinary action. Similarly, emailing with or otherwise communicating with other students or anyone else during a quiz or exam will be treated as a violation and also referred for disciplinary action.   Cheating and plagiarism have been and will be strongly penalized. Please review academic integrity policies [here](http://academicintegrity.ucsd.edu).

You are responsible for ensuring that the correct file has been submitted and that the submission is uncorrupted. If, for whatever reason, Canvas or DataHub is down or something else prohibits you from being able to turn in an assignment on time, immediately contact the instructor by emailing the assignment, otherwise the assignment will be graded as late.


### Class Conduct

In all interactions in this class, you are expected to be respectful. This includes following the [UC San Diego principles of community](https://ucsd.edu/about/principles.html).
 
This class will be a welcoming, inclusive, and harassment-free experience for everyone, regardless of gender, gender identity and expression, age, sexual orientation, disability, physical appearance, body size, race, ethnicity, religion (or lack thereof), political beliefs/leanings, or technology choices.

At all times, you should be considerate and respectful. Always refrain from demeaning, discriminatory, or harassing behavior and speech. Last of all, take care of each other.

If you have a concern, please speak with the Professor, your TAs, or IAs. If you are uncomfortable doing so, that's OK! The [OPHD](https://blink.ucsd.edu/HR/policies/sexual/OPHD.html) (Office for the Prevention of Sexual Harassment and Discrimination) and [CARE](https://care.ucsd.edu/) (confidential advocacy and education office for sexual violence and gender-based violence) are wonderful resources on campus.  


### Disability Access

Students requesting accommodations due to a disability must provide a current Authorization for Accommodation (AFA) letter. These letters are issued by the Office for Students with Disabilities (OSD), which is located in University Center 202 behind Center Hall. Please make arrangements to contact Professor privately to arrange accommodations.

Contacting the OSD can help you further:  
858.534.4382 (phone)  
osd@ucsd.edu (email)  
http://disabilities.ucsd.edu  

### Important Resources for Students

* [UCSD’s principles of community](https://ucsd.edu/about/principles.html)

* [Counseling and Psychology Services (CAPS)](https://wellness.ucsd.edu/CAPS/Pages/default.aspx).  “CAPS provides FREE, confidential, psychological counseling and crisis services for registered UCSD students.  CAPS also provides a variety of groups, workshops, and drop-in forums.”

* [CARE at the Sexual Assault Resource Center](https://care.ucsd.edu/) is the UC San Diego confidential advocacy and education office for sexual harassment, sexual violence and gender-based violence (dating violence, domestic violence, stalking). 

* [Office for the Prevention of Harassment & Discrimination (OPHD)](https://ophd.ucsd.edu/).  OPHD "works to resolve complaints of discrimination and harassment through formal investigation or alternative resolution."
  

