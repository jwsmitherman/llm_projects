**To:** April Davis
**Cc:** Teresa; Jasmine; Pod 2 team
**Subject:** USAJOBS analysis: key findings for the Cycle 4 health and admin series (2024–2026)

Hi April,

I've finished the first full pass of the USAJOBS analysis. It covers announcements opened in 2024, 2025 and 2026 (through mid-November) for nine series: 0560, 0661, 0667, 0669, 0671, 0672, 0673, 0950 and 1103. All data comes from the USAJOBS Historic JOA API, including the full announcement text. The flat files are ready for the SharePoint data folder ("USA Jobs 2024/2025/2026 data"). Each series also has its own workbook with the summary, organizational titles, grade distribution, agency comparison, keyword evidence and education signals.

Here are the findings most relevant to what the pods have been asking.

### 1. The 2025 freeze dip is clear, and most series have partly recovered in 2026

| Series | 2024 | 2025 | 2026 (YTD) |
|---|---|---|---|
| 0560 Budget Analysis | 3,950 | 843 | 1,821 |
| 0671 Health System Specialist | 2,048 | 1,020 | 1,625 |
| 0661 Pharmacy Technician | 1,273 | 862 | 981 |
| 0950 Paralegal Specialist | 1,052 | 320 | 885 |
| 0669 Medical Records Administration | 203 | 178 | **263** |
| 0672 Prosthetic Representative | 169 | 156 | **181** |
| 0667 Orthotist & Prosthetist | 157 | 120 | 117 |
| 0673 Hospital Housekeeping Management | 127 | 67 | 109 |
| 1103 Industrial Property Management | 108 | 29 | 25 |

2024 remains the best full-year baseline. 0669 and 0672 have already passed their 2024 volume. 1103 hasn't recovered, so its 2025–26 figures rest on very small samples.

### 2. VA drives the posting data for the health series

The VA share of announcements, 2024–2026, is 100% for 0672, 93–99% for 0667, 81–89% for 0661, 77–94% for 0673, 65–78% for 0669 and 65–71% for 0671. As a result, the USAJOBS picture for these series is mostly a VA picture. When we read grade patterns against the OPM Title 5 standard, we should look at VA and non-VA postings separately. Each workbook has an `agency_comparison` sheet that does this split.

### 3. 0661: the higher grades come from VA's Title 38 postings (Shauna's question)

- **VA:** about 98% of VA 0661 postings cite Title 38. Their most common entry grade is **GS-8 in 2024 and GS-7 in 2025–26**, with a median starting salary of about $61K.
- **Non-VA (DoD, HHS, DOJ):** the most common entry grade and full performance level are both **GS-6**, with a median starting salary of about $50K.
- **Takeaway:** the grades above the GS-5/6 range come from VA's own Title 38 qualification standard, not from a change in the work across government. This supports Shauna's read that the current grade structure still fits the Title 5 work.

### 4. 0667, 0669, 0671 and 0673 show the same VA-versus-everyone-else gap

- **0667:** nearly every posting is a VA posting citing Title 38, entering at **GS-12** (about $88K). Non-VA had 8 or fewer postings a year, at GS-7 to GS-9. Degree or basic-requirement language appears in about 95% of postings, including AAOP/IAOP and ABC credentialing language. This confirms Carly's point that VA's own qualification standard drives what we see.
- **0669:** VA enters at **GS-12** and non-VA at **GS-9**. Duty language around clinical documentation and coding is becoming more common, and AHIMA certification language appears often.
- **0671:** VA enters mainly at **GS-12** and non-VA at **GS-9**. Most VA 0671 postings don't cite Title 38. Language tying the role to health system administration or the medical center director appears in **43–48%** of postings. That supports bringing the 1970s 0670/0671 guidance into the standard.
- **0673:** VA postings enter at GS-12 to GS-13, about $90–95K.

### 5. 0672 changed how it hires in 2026

In 2026, **72 of 181** postings opened at **GS-5**, mostly Prosthetic Representative and Technical Career Field (TCF) trainee career-ladder postings. In 2024–25 the most common entry grade was GS-12. This is worth checking with the classifier on 0672 before finalizing the grade-level descriptions.

### 6. Checking the AI redline recommendations against real postings

- **1103:** "property disposal" and "life cycle" language rose from **18% (2024) to 32% (2026)** of postings. "Asset management" went from 0% to 7–8%. Both support the redline recommendations. "Supply chain" appeared in only one posting, so it isn't supported by the data. Note the sample is small: 25 postings in 2026.
- **0950:** legal technology platforms (Westlaw, Lexis, case management systems) appear in **14–17%** of postings, which supports modernizing the knowledge language. E-discovery/ESI (about 3–5%) and information governance (about 2%) appear, but rarely. They may belong in the occupational information rather than in the core factor language. DOJ's share of postings grew from 24% in 2024 to **57%** in 2026, so the 2026 language leans toward U.S. Attorneys' offices.
- **0560:** financial-system references (ERP, Oracle, G-Invoicing and similar) appear in **12–15%** of postings.

### 7. Skills-based hiring is visible in the postings

The share of postings that mention a structured assessment rose sharply from 2024 to 2026. Assessments counted include SME-QA, structured interviews, work samples, USA Hire and technical assessments.

- **0950:** 35% → 78%
- **0560:** 17% → 44%
- **1103:** 7% → 40%
- **0673:** 66% → 84%

The health series with education or licensure requirements still carry basic-requirement language in 80–100% of postings: 0661, 0667, 0669 and 0672. That fits their individual occupational requirements and isn't a gap.

### Caveats, and what I'm fixing next

- **"AI" mentions are mostly boilerplate.** Most hits are application warnings like "do not use AI tools to write your responses." They don't describe the work, so I'll exclude them before we cite technology trends.
- **"Automation" for 0661 (85–91%) is real.** It refers to automated dispensing machines and pharmacy systems, which is useful for the Impact of Automation section.
- **Three fields came back empty: hiring paths, telework and multi-series.** The API names those nested fields differently than its documentation. I'll map them correctly.
- **Some grade extremes are artifacts.** Values like grade 0, 28 or 52 come from non-GS pay plans such as NH, CN and ES. I'll limit grade statistics to GS/GG.
- **Competency extraction and the emerging-language view picked up too much boilerplate.** Examples are financial disclosure notices, PCS/relocation text and "skills and abilities" fragments. In the next pass I'll limit both to the duties and qualifications sections and filter out standard announcement language. That will make the technical-competency candidates usable for the IOs.
- **Postings aren't the same as on-board employees.** These numbers show recruiting activity, so we should read them alongside EHRI before choosing illustration grade levels.

If it helps, I can walk the pods through their series workbooks at the next pod meeting. I can also add any other series or recommendation terms you want checked.

Thanks,
Josh
