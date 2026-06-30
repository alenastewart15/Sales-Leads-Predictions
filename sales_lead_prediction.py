#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 21:41:55 2026

@author: alenastewart
"""

# Final ITM251 Project
# Created by Eva Bollin, Alena Stewart, Natalie Quear
# Using ML, Automation, & Data Analysis for Sales

import pandas as pd 
# AI and research recommended this library for data in Excel
import matplotlib.pyplot as plt
# Used in class for graphs
from sklearn.model_selection import train_test_split
# This converts text into numbers so it's usable for things like linear regression in machine learning or the RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
# For the machine learning...uses patterns in the data to make predictions!

# Opening & Reading the Dataset
leads = pd.read_csv('/Users/alenastewart/Desktop/ITM 251/leads_dataset.csv')

# Data Analysis & Visual Graphs
print("-" * 30)
print("| Sales Leads Analysis 💵📊 |")
print("-" * 30)
print()
print("Data Analysis & Important Graphs:")
print("-" * 90)
print()
hot = 0
warm = 0
cold = 0

hot_total = 0
warm_total = 0
cold_total = 0

for index, row in leads.iterrows(): #leads.iterrows() goes through each row in the DataFrame
    value = int(row['estimated_order_value'])
    if row['lead_temperature'] == "Hot":
        hot = hot + 1
        hot_total = hot_total + value
    elif row['lead_temperature'] == "Warm":
        warm = warm + 1
        warm_total = warm_total + value
    elif row['lead_temperature'] == "Cold":
        cold = cold + 1
        cold_total = cold_total + value

categories = ["Hot", "Warm", "Cold"]
counts = [hot, warm, cold]

plt.bar(categories, counts, color = ['hotpink', 'coral', 'lightblue'])
plt.title("Lead Temperature Categorization")
plt.xlabel("Lead Category")
plt.ylabel("Number of Leads")
plt.show()
print("🟥 Hot Leads:", hot)
print("🟧 Warm Leads:", warm)
print("🟦 Cold Leads:", cold)
print()

hot_avg = hot_total / hot
warm_avg = warm_total / warm
cold_avg = cold_total / cold
averages = [hot_avg, warm_avg, cold_avg]

plt.bar(categories, averages, color = ['hotpink', 'coral', 'lightblue'])
plt.title("Average Order Value by Lead Category")
plt.xlabel("Lead Category")
plt.ylabel("Average Order Value in $")
plt.show()
print("🟥 Hot Leads Value Average: $" + str(round(hot_avg, 2)))
print("🟧 Warm Leads Value Average: $" + str(round(warm_avg, 2)))
print("🟦 Cold Leads Value Average: $" + str(round(cold_avg, 2)))
print()

print("High Priority Leads to Contact (Events Soon! 📆):")
print("These are leads with events in the next 14 days, a contact reached, a confirmed budget, and no merch ordered in the last 4 months:")
print()
for index, row in leads.iterrows():
    if 0 <= int(row["event_days_away"]) <= 14 and row["decision_maker_contacted"] == "Yes" and row["budget_confirmed"] == "Yes" and int(row["last_merch_purchase_months_ago"]) > 4:
        print("- " + row['company_name'])
        
print()

# Machine Learning Section
# One of the biggest challenges in sales is constant rejection!
# Will this lead have a high success rate if pursued?

# Convert values to int with Pandas since int(leads['estimated_order_value']) gets error
leads['estimated_order_value'] = leads['estimated_order_value'].astype(int)

# Uses median to split data
threshold = leads['estimated_order_value'].median()

# Creates classification so the computer has something to learn from
# 1 = high value lead
# 0 = low value lead
leads['high_success_prediction'] = leads['estimated_order_value'].apply(lambda x: 1 if x > threshold else 0)

print()
print("Machine Learning Balance Check:")
print("-" * 90)
print()
print("Classification Balance (0 = low value, 1 = high value):")
print(leads['high_success_prediction'].value_counts())
print()

prediction_inputs = ['industry', 'company_size', 'lead_source']
x = leads[prediction_inputs]
x = pd.get_dummies(x)
y = leads['high_success_prediction']

# 80% training (model learns)
# 20% testing (model gets evaluated)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)

model = RandomForestClassifier(class_weight='balanced', random_state = 1)
model.fit(x_train, y_train)

predictions = model.predict(x_test)

correct = (predictions == y_test).sum()
total = len(y_test)
accuracy = round(correct / total * 100, 2)

print()
leads['predicted_success'] = model.predict(x)
leads['success_probability'] = model.predict_proba(x)[:, 1]

# Conclusion Dashboard
def print_dashboard(leads):

    print("Leads Success Prediction Dashboard:")
    print("-" * 90)

    print()
    print("Machine Learning Model Accuracy:", accuracy, "%")
    if accuracy > .7:
        print("Good Accuracy! ✅")
    else:
        print("Bad Accuracy! ❌")

    total_leads = len(leads)
    successful_leads = (leads['predicted_success'] == 1).sum()

    print()
    print("Lead Summary:")
    print("Total Leads:", total_leads)
    print("Successful Leads:", successful_leads)
    print("Unsuccessful Leads:", (total_leads - successful_leads))

    avg_all = leads['estimated_order_value'].mean()
    successful_subset = leads[leads['predicted_success'] == 1]
    avg_successful = successful_subset['estimated_order_value'].mean()

    print()
    print("Revenue Information:")
    print("Average Order Value (All Leads): $", round(avg_all, 2))
    print("Average Order Value (Successful Leads): $", round(avg_successful, 2))

    conversion_rate = (successful_leads / total_leads) * 100
    print()
    print("Successful Lead Rate (Conversion):", round(conversion_rate, 2), "%")

    print()
    print("Top 5 Leads to Pursue:")
    top5 = leads.sort_values(by = 'estimated_order_value', ascending = False).head(5)
    for i, row in top5.iterrows():
        print("-", row['company_name'], "| $", row['estimated_order_value'], "|", row['email_address'])

    print()
    print("Sample Predictions:")
    sample = leads[['company_name', 'estimated_order_value', 'predicted_success', 'success_probability']].head(5)
    for i, row in sample.iterrows():
        if row['predicted_success'] == 1:
            label = "LIKELY SUCCESS 🎉"
        else:
            label = "UNLIKELY SUCCESS"
            
        prob_percent = round(row['success_probability'] * 100, 2)
        print("-", row['company_name'], "=", label, "|", prob_percent, "% confidence")
        
    plt.hist(leads['success_probability'], bins = 10, color = ["darkcyan"])
    plt.title("Distribution of Lead Success Probability")
    plt.xlabel("Probability of Success")
    plt.ylabel("Number of Leads")
    plt.show()

print_dashboard(leads)

# Email Automation (Since this is a project, we will not actually be sending real emails...)
def generate_email(lead):
    
    if lead['predicted_success'] == 1:
        print("Subject: Custom Merchandise Opprotunity for", lead["upcoming_event"])
        print()
        print("Good Afternoon " + lead['company_name'] + ",")
        print()
        print("We noticed your company has an upcoming event, and we would love to help provide you with custom merchandise for it! We would love to chat about what we can do for you or your business for merchandise or promotional products, do you have time to chat this week?")
        print()
        print("Best Regards,")
        print("Sales Team")
    else:
        print("Subject: Custom Merchandise for", lead["company_name"])
        print()
        print("Good Afternoon " + lead['company_name'] + ",")
        print()
        print("We’d love to connect and learn more about your business needs for custom merchandise. We offer a variety of  options that could be a great fit including tshirts, hats, bags, and more. We look foward to hearing from you!")
        print()
        print("Best Regards,")
        print("Sales Team")

def show_sample_emails(leads):
    print()
    print()
    print("Automated Priority Emails:")
    print("-" * 90)
    print()
    
    successful = leads[leads['predicted_success'] == 1].to_dict('records')
    unsuccessful = leads[leads['predicted_success'] == 0].to_dict('records')

    print("Successful Prediction Lead Email Samples:")
    print("-" * 40)
    for lead in successful[:2]:
        generate_email(lead)
        print("-" * 40)

    print()
    print("Unuccessful Prediction Lead Email Samples:")
    print("-" * 40)
    for lead in unsuccessful[:2]:
        generate_email(lead)
        print("-" * 40)

show_sample_emails(leads)
