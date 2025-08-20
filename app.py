from flask import Flask,render_template, request
import joblib
import numpy as np
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

app = Flask(__name__)

# Load trained model and label encoder
model = joblib.load("best_diet_model.pkl")
le = joblib.load("label_Encoder.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    diet_plan = {}
    workout_plan = {}
    bmi_value = None
    bmi_category = None

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
            
            # Determine BMI category
            if bmi_value < 18.5:
                bmi_category = "Underweight"
            elif 18.5 <= bmi_value < 25:
                bmi_category = "Normal weight"
            elif 25 <= bmi_value < 30:
                bmi_category = "Overweight"
            else:
                bmi_category = "Obese"

            # Prepare features (must match training data)
            diet_mapping = {'Vegetarian': 0, 'Non-Vegetarian': 1, 'Vegan': 2}
            exercise_mapping = {'Cardio': 0, 'Strength Training': 1, 'Yoga': 2}
            
            features = np.array([[
                age,
                height_cm,
                weight,
                1 if gender == 'Male' else 0,
                diet_mapping[diet_pref],
                exercise_mapping[exercise_pref]
            ]])

            # Make prediction
            pred_encoded = model.predict(features)[0]
            prediction = le.inverse_transform([pred_encoded])[0]

            # Generate recommendations
            diet_plan = generate_diet_plan(prediction, diet_pref)
            workout_plan = generate_workout_plan(prediction, exercise_pref)

        except Exception as e:
            print(f"Error: {str(e)}")

    return render_template("index.html",
                         prediction=prediction,
                         diet_plan=diet_plan,
                         workout_plan=workout_plan,
                         bmi_value=bmi_value,
                         bmi_category=bmi_category)

def generate_diet_plan(bmi_category, diet_pref):
    plans = {
        'Underweight': {
            'Vegetarian': {
                'morning': ['Banana smoothie with almond butter (500 cal)', 'Whole grain toast with avocado', 'Full-fat yogurt with honey'],
                'afternoon': ['Paneer curry with brown rice', 'Mixed vegetable salad with olive oil dressing', 'Lentil soup with bread'],
                'evening': ['Khichdi with ghee', 'Handful of mixed nuts', 'Milk with turmeric before bed']
            },
            'Non-Vegetarian': {
                'morning': ['3-egg omelet with cheese', 'Whole wheat toast with peanut butter', 'Glass of whole milk'],
                'afternoon': ['Grilled chicken with quinoa', 'Steamed vegetables with olive oil', 'Fish curry with rice'],
                'evening': ['Salmon with sweet potato mash', 'Handful of almonds', 'Protein shake']
            },
            'Vegan': {
                'morning': ['Smoothie with banana, oats, and almond milk', 'Avocado toast with chia seeds', 'Handful of walnuts'],
                'afternoon': ['Quinoa bowl with roasted vegetables', 'Lentil curry with whole wheat roti', 'Chickpea salad'],
                'evening': ['Tofu stir-fry with brown rice', 'Nut butter with whole grain crackers', 'Soy milk with dates']
            }
        },
        'Normal weight': {
            'Vegetarian': {
                'morning': ['Oatmeal with berries and nuts', 'Whole grain toast with almond butter', 'Green tea'],
                'afternoon': ['Vegetable stir-fry with tofu', 'Brown rice with dal', 'Yogurt with fruits'],
                'evening': ['Vegetable soup with whole wheat bread', 'Sprouts salad', 'Herbal tea']
            },
            'Non-Vegetarian': {
                'morning': ['Scrambled eggs with spinach', 'Whole grain toast', 'Green smoothie'],
                'afternoon': ['Grilled chicken with roasted vegetables', 'Quinoa salad', 'Buttermilk'],
                'evening': ['Baked fish with steamed veggies', 'Sweet potato mash', 'Handful of nuts']
            },
            'Vegan': {
                'morning': ['Chia pudding with almond milk', 'Whole grain toast with hummus', 'Fruit salad'],
                'afternoon': ['Lentil soup with whole wheat bread', 'Roasted vegetable quinoa bowl', 'Green salad'],
                'evening': ['Stir-fried tofu with vegetables', 'Brown rice', 'Herbal tea']
            }
        },
        'Overweight': {
            'Vegetarian': {
                'morning': ['Vegetable smoothie (spinach, cucumber, lemon)', 'Handful of soaked almonds', 'Green tea'],
                'afternoon': ['Dal with multigrain roti', 'Vegetable salad with lemon dressing', 'Buttermilk'],
                'evening': ['Vegetable soup', 'Sprouts salad', 'Herbal tea']
            },
            'Non-Vegetarian': {
                'morning': ['Boiled eggs (2 whites, 1 whole)', 'Steamed vegetables', 'Green tea'],
                'afternoon': ['Grilled chicken breast', 'Quinoa or brown rice', 'Steamed vegetables'],
                'evening': ['Baked fish with lemon', 'Vegetable salad', 'Clear soup']
            },
            'Vegan': {
                'morning': ['Green smoothie (kale, apple, lemon)', 'Handful of soaked chia seeds', 'Herbal tea'],
                'afternoon': ['Lentil soup', 'Steamed vegetables', 'Quinoa'],
                'evening': ['Stir-fried tofu with minimal oil', 'Vegetable salad', 'Herbal tea']
            }
        },
        'Obese': {
            'Vegetarian': {
                'morning': ['Warm lemon water', 'Vegetable juice (no sugar)', 'Handful of soaked almonds'],
                'afternoon': ['Steamed vegetables with dal', 'Multigrain roti (1 small)', 'Buttermilk'],
                'evening': ['Clear vegetable soup', 'Sprouts salad (no dressing)', 'Herbal tea']
            },
            'Non-Vegetarian': {
                'morning': ['Boiled egg whites (2)', 'Cucumber slices', 'Green tea'],
                'afternoon': ['Grilled chicken (skinless)', 'Steamed vegetables', 'Clear soup'],
                'evening': ['Baked fish (small portion)', 'Lettuce salad (no dressing)', 'Warm water with lemon']
            },
            'Vegan': {
                'morning': ['Detox water', 'Handful of soaked almonds', 'Herbal tea'],
                'afternoon': ['Steamed vegetables with minimal salt', 'Lentil soup (clear)', 'Herbal tea'],
                'evening': ['Raw vegetable salad (no dressing)', 'Clear vegetable broth', 'Warm water with lemon']
            }
        }
    }
    
    return plans.get(bmi_category, {}).get(diet_pref, {
        'morning': ['Consult nutritionist for personalized plan'],
        'afternoon': ['Consult nutritionist for personalized plan'],
        'evening': ['Consult nutritionist for personalized plan']
    })
def generate_workout_plan(bmi_category, exercise_pref):
    plans = {
        'Cardio': {
            'Underweight': {
                'morning': ['15 min light jogging', '10 min stretching'],
                'afternoon': ['30 min swimming (moderate pace)', '10 min rest'],
                'evening': ['20 min brisk walking', 'Breathing exercises']
            },
            'Normal weight': {
                'morning': ['20 min jogging', '5 min dynamic stretches'],
                'afternoon': ['30 min cycling', '10 min core exercises'],
                'evening': ['15 min jump rope', 'Cool down stretches']
            },
            'Overweight': {
                'morning': ['30 min brisk walking', '5 min warm-up'],
                'afternoon': ['20 min elliptical trainer', '10 min light stretching'],
                'evening': ['15 min water aerobics', 'Breathing exercises']
            },
            'Obese': {
                'morning': ['20 min slow walking', '5 min gentle stretching'],
                'afternoon': ['15 min chair exercises', '10 min rest'],
                'evening': ['10 min standing stretches', 'Deep breathing']
            }
        },
        'Strength Training': {
            'Underweight': {
                'morning': ['Bodyweight exercises (3 sets)', 'Push-ups (10 reps)'],
                'afternoon': ['Light dumbbell workout (3kg)', 'Squats (15 reps)'],
                'evening': ['Resistance band exercises', 'Stretching']
            },
            'Normal weight': {
                'morning': ['Weight training (upper body)', 'Pull-ups (3 sets)'],
                'afternoon': ['Squats with weights (3 sets)', 'Deadlifts (3 sets)'],
                'evening': ['Core exercises', 'Cool down stretches']
            },
            'Overweight': {
                'morning': ['Bodyweight squats (15 reps)', 'Wall push-ups (10 reps)'],
                'afternoon': ['Resistance band workout', 'Chair dips (10 reps)'],
                'evening': ['Light core exercises', 'Stretching']
            },
            'Obese': {
                'morning': ['Chair-assisted squats (10 reps)', 'Arm circles (2 min)'],
                'afternoon': ['Seated leg lifts (10 reps)', 'Shoulder rolls'],
                'evening': ['Gentle stretching', 'Breathing exercises']
            }
        },
        'Yoga': {
            'Underweight': {
                'morning': ['Surya Namaskar (5 rounds)', 'Balancing poses'],
                'afternoon': ['Standing asanas', 'Pranayama'],
                'evening': ['Restorative yoga', 'Meditation (10 min)']
            },
            'Normal weight': {
                'morning': ['Surya Namaskar (10 rounds)', 'Core-focused asanas'],
                'afternoon': ['Hip-opening poses', 'Breathwork'],
                'evening': ['Forward bends', 'Yoga Nidra']
            },
            'Overweight': {
                'morning': ['Surya Namaskar (5 modified rounds)', 'Seated twists'],
                'afternoon': ['Supported standing poses', 'Pranayama'],
                'evening': ['Gentle stretching', 'Guided meditation']
            },
            'Obese': {
                'morning': ['Chair yoga (10 min)', 'Breathing exercises'],
                'afternoon': ['Seated stretches', 'Guided relaxation'],
                'evening': ['Deep breathing', 'Mindfulness practice']
            }
        }
    }
    
    return plans.get(exercise_pref, {}).get(bmi_category, {
        'morning': ['Consult trainer for personalized plan'],
        'afternoon': ['Consult trainer for personalized plan'],
        'evening': ['Consult trainer for personalized plan']
    })

if __name__ == "__main__":
    app.run(debug=True)