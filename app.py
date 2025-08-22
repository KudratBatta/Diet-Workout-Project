from flask import Flask, render_template, request, redirect, url_for, session
import joblib
import numpy as np
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

app = Flask(__name__)
app.secret_key = "yoursecretkey"  # Needed for session storage

# Load trained model and label encoder
model = joblib.load("best_diet_model.pkl")
le = joblib.load("label_Encoder.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            # Get form data
            age = int(request.form['age'])
            gender = request.form['gender']
            height_cm = float(request.form['height'])
            weight = float(request.form['weight'])
            diet_pref = request.form['diet']
            exercise_pref = request.form['exercise']

            # Calculate BMI
            height_m = height_cm / 100
            bmi_value = round(weight / (height_m ** 2), 2)

            if bmi_value < 18.5:
                bmi_category = "Underweight"
            elif 18.5 <= bmi_value < 25:
                bmi_category = "Normal weight"
            elif 25 <= bmi_value < 30:
                bmi_category = "Overweight"
            else:
                bmi_category = "Obese"

            # Prepare features
            diet_mapping = {'Vegetarian': 0, 'Non-Vegetarian': 1, 'Vegan': 2}
            exercise_mapping = {'Cardio': 0, 'Strength Training': 1, 'Yoga': 2}

            features = np.array([[age, height_cm, weight,
                                  1 if gender == 'Male' else 0,
                                  diet_mapping[diet_pref],
                                  exercise_mapping[exercise_pref]]])

            # Make prediction
            pred_encoded = model.predict(features)[0]
            prediction = le.inverse_transform([pred_encoded])[0]

            # Store results and form data in session
            session['results'] = {
                "prediction": prediction,
                "diet_plan": generate_diet_plan(prediction, diet_pref),
                "workout_plan": generate_workout_plan(prediction, exercise_pref),
                "bmi_value": bmi_value,
                "bmi_category": bmi_category,
                "diet": diet_pref,
                "exercise": exercise_pref,
                "form_data": {
                    "age": age,
                    "gender": gender,
                    "height": height_cm,
                    "weight": weight,
                    "diet": diet_pref,
                    "exercise": exercise_pref
                }
            }

            return redirect(url_for("index"))

        except Exception as e:
            print(f"Error: {str(e)}")

    results = session.pop("results", None)  # Clear after showing once
    return render_template("index.html", **(results or {}))


def generate_diet_plan(bmi_category, diet_pref):
    plans = {
        'Underweight': {
            'Vegetarian': {
                'morning': ['Oats with nuts and seeds + a glass of milk', 'Sprouts salad + paneer cubes'],
                'afternoon': ['Quinoa with mixed vegetables + lentil soup', '2 whole wheat rotis + dal + sabzi'],
                'evening': ['Sweet potato salad + tofu', 'Peanut butter sandwich + a banana']
            },
            'Non-Vegetarian': {
                'morning': ['Scrambled eggs with cheese + a glass of milk', 'Chicken and vegetable omelet'],
                'afternoon': ['Grilled chicken breast with brown rice and steamed veggies', 'Fish curry with whole wheat roti'],
                'evening': ['Turkey sandwich on whole wheat bread', 'Chicken and bean soup']
            },
            'Vegan': {
                'morning': ['Avocado toast on whole grain bread + a plant-based smoothie', 'Oats with nut butter, berries, and chia seeds'],
                'afternoon': ['Lentil soup with quinoa and roasted vegetables', 'Tofu scramble with spinach and mushrooms'],
                'evening': ['Chickpea curry with brown rice', 'Veggie burger with sweet potato fries']
            }
        },
        'Normal weight': {
            'Vegetarian': {
                'morning': ['Vegetable poha + a glass of skim milk', 'Fruit salad with yogurt and a handful of almonds'],
                'afternoon': ['Dal tadka + brown rice + salad', 'Multigrain roti + paneer sabzi + soup'],
                'evening': ['Stir-fried vegetables with tofu', 'Lentil soup with a side of steamed vegetables']
            },
            'Non-Vegetarian': {
                'morning': ['Boiled eggs + whole wheat toast', 'Yogurt with berries and granola'],
                'afternoon': ['Grilled fish with quinoa and green beans', 'Chicken salad with light dressing'],
                'evening': ['Baked salmon with roasted asparagus and potatoes', 'Lean steak with a large green salad']
            },
            'Vegan': {
                'morning': ['Spinach and mushroom smoothie + a piece of fruit', 'Oatmeal with berries and a sprinkle of nuts'],
                'afternoon': ['Black bean burger on a whole wheat bun', 'Quinoa bowl with roasted veggies and a tahini dressing'],
                'evening': ['Vegetable curry with a side of brown rice', 'Stir-fried tofu with mixed vegetables']
            }
        },
        'Overweight': {
            'Vegetarian': {
                'morning': ['Green smoothie with spinach and a scoop of protein powder', 'Oats with berries and seeds'],
                'afternoon': ['Large vegetable salad with chickpeas and light dressing', 'Sprouts salad + a cup of lentil soup'],
                'evening': ['Grilled paneer with bell peppers and onions', 'Mixed vegetable stir-fry']
            },
            'Non-Vegetarian': {
                'morning': ['Scrambled egg whites with spinach and mushrooms', 'Greek yogurt with a handful of berries'],
                'afternoon': ['Large chicken salad with a variety of greens and a light vinaigrette', 'Tuna salad (made with Greek yogurt) on lettuce wraps'],
                'evening': ['Baked cod with steamed broccoli and lemon', 'Lean turkey stir-fry with a low-sodium sauce']
            },
            'Vegan': {
                'morning': ['Chia seed pudding made with almond milk and berries', 'Tofu scramble with a side of fresh fruit'],
                'afternoon': ['Lentil soup with a side of mixed greens', 'Quinoa bowl with a variety of raw vegetables'],
                'evening': ['Steamed edamame with a side of cauliflower rice', 'Large green salad with grilled portobello mushrooms']
            }
        },
        'Obese': {
            'Vegetarian': {
                'morning': ['High-fiber cereal with skim milk', 'A large fruit bowl with a sprinkle of nuts'],
                'afternoon': ['Vegetable soup with a small salad', 'Grilled vegetables with a small portion of brown rice'],
                'evening': ['Cauliflower rice bowl with grilled tofu and a variety of low-carb vegetables', 'A large green salad with grilled paneer']
            },
            'Non-Vegetarian': {
                'morning': ['Boiled egg whites + a small piece of fruit', 'Low-fat Greek yogurt with berries'],
                'afternoon': ['Grilled chicken breast with steamed greens', 'Tuna salad (no mayo) on a bed of lettuce'],
                'evening': ['Baked fish with roasted vegetables', 'Lean turkey or chicken breast with a large green salad']
            },
            'Vegan': {
                'morning': ['Spinach and berry smoothie with plant-based protein powder', 'A small bowl of oatmeal with a few berries'],
                'afternoon': ['Large salad with kidney beans and a light dressing', 'Broccoli and mushroom stir-fry'],
                'evening': ['Steamed vegetables with a side of lentils', 'Black bean soup with a side of mixed greens']
            }
        }
    }
    return plans.get(bmi_category, {}).get(diet_pref, {})


def generate_workout_plan(bmi_category, exercise_pref):
    plans = {
        'Underweight': {
            'Cardio': {
                'morning': ['20 min brisk walk', '15 min jogging'],
                'afternoon': ['15 min cycling', '10 min swimming'],
                'evening': ['15 min elliptical training', '10 min light cardio']
            },
            'Strength Training': {
                'morning': ['3 sets of 10 push-ups', '3 sets of 15 squats'],
                'afternoon': ['3 sets of 12 bicep curls (light weights)', '3 sets of 10 lunges'],
                'evening': ['3 sets of 10 pull-ups (assisted if needed)', '3 sets of 15 tricep dips']
            },
            'Yoga': {
                'morning': ['Warrior Pose Flow (Virabhadrasana)', 'Sun Salutations'],
                'afternoon': ['Chair Pose (Utkatasana) and Tree Pose (Vrksasana)'],
                'evening': ['Restorative Yoga Poses (e.g., Supported Bridge Pose)']
            }
        },
        'Normal weight': {
            'Cardio': {
                'morning': ['30 min running', '45 min cycling'],
                'afternoon': ['30 min swimming', '20 min HIIT'],
                'evening': ['30 min elliptical training', '45 min brisk walk']
            },
            'Strength Training': {
                'morning': ['Full body workout (bench press, squats, deadlifts)', 'Upper body workout (bicep curls, pull-ups)'],
                'afternoon': ['Lower body workout (lunges, leg press)', 'Core workout (planks, crunches)'],
                'evening': ['Circuit training with moderate weights', '3 sets of 15 kettlebell swings']
            },
            'Yoga': {
                'morning': ['Vinyasa Flow', 'Power Yoga session'],
                'afternoon': ['Ashtanga Yoga practice', 'Balancing poses (e.g., Crow Pose)'],
                'evening': ['Hatha Yoga session', 'Stretching and flexibility poses']
            }
        },
        'Overweight': {
            'Cardio': {
                'morning': ['45 min brisk walk', '30 min stationary bike'],
                'afternoon': ['30 min elliptical training', '45 min swimming'],
                'evening': ['45 min jogging or power walking', '30 min stair climbing machine']
            },
            'Strength Training': {
                'morning': ['3 sets of 15 bodyweight squats', '3 sets of 10 push-ups (on knees if needed)'],
                'afternoon': ['3 sets of 20 lunges', '3 sets of 12 rows with resistance bands'],
                'evening': ['Circuit training with light weights and high reps', 'Bodyweight exercises (planks, glute bridges)']
            },
            'Yoga': {
                'morning': ['Beginner-friendly Vinyasa flow', 'Gentle Yoga for Flexibility'],
                'afternoon': ['Chair Yoga for stability and balance', 'Restorative Yoga with props'],
                'evening': ['Slow-paced Hatha Yoga', 'Stretching routine for hips and back']
            }
        },
        'Obese': {
            'Cardio': {
                'morning': ['60 min brisk walking (low impact)', '45 min cycling (light resistance)'],
                'afternoon': ['45 min swimming laps (gentle pace)', '30 min elliptical (low incline)'],
                'evening': ['45 min walking on a treadmill', '20 min low-impact aerobics']
            },
            'Strength Training': {
                'morning': ['Bodyweight squats (3 sets of 10)', 'Wall push-ups (3 sets of 12)'],
                'afternoon': ['Seated resistance band rows (3 sets of 15)', 'Chair-assisted lunges (3 sets of 10)'],
                'evening': ['Planks (start with 20 seconds, gradually increase)', 'Glute bridges (3 sets of 15)']
            },
            'Yoga': {
                'morning': ['Gentle stretching and mobility exercises', 'Chair Yoga'],
                'afternoon': ['Yin Yoga for deep stretching', 'Basic Yoga Poses (e.g., Mountain Pose, Cat-Cow Pose)'],
                'evening': ['Restorative Yoga to release tension', 'Breathing exercises and meditation']
            }
        }
    }
    return plans.get(bmi_category, {}).get(exercise_pref, {})


if __name__ == "__main__":
    app.run(debug=True)