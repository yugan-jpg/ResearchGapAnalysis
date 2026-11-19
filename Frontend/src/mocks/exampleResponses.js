// Mock-only presentation data. It is never used by production API calls.
export const examplePapers = [
  { id: 'example-1', filename: 'Adaptive control in ecological systems.pdf', status: 'ready', title: 'Adaptive control in ecological systems', authors: ['R. Chen', 'M. Iqbal'], year: 2024, pages: 18 },
  { id: 'example-2', filename: 'Reproducibility of low-data scientific models.pdf', status: 'processing', title: 'Reproducibility of low-data scientific models', authors: ['S. Alvarez'], year: 2023, pages: 11 },
];

export const exampleAnalysis = {
  answer: 'The supplied papers indicate promising adaptive methods, but the evidence base remains narrow and inconsistently evaluated.',
  evidence_sufficient: true,
  sections: {
    methodologies: [{ claim: 'Most studies use retrospective simulations with a single benchmark dataset.', citations: [{ paper_id: 'example-1', source: 'Adaptive control in ecological systems', page: 6, excerpt: 'We evaluated the controller retrospectively on one regional observation set.' }] }],
    findings: [{ claim: 'Adaptive controllers improved short-horizon accuracy in reported experiments.', citations: [{ paper_id: 'example-1', source: 'Adaptive control in ecological systems', page: 9, excerpt: 'Accuracy gains were strongest in the first three forecast intervals.' }] }],
    limitations: [{ claim: 'External validation and prospective evaluation are limited.', citations: [{ paper_id: 'example-2', source: 'Reproducibility of low-data scientific models', page: 7, excerpt: 'Results were sensitive to the split strategy and did not replicate across all seeds.' }] }],
    contradictions: [{ claim: 'Reported gains differ on whether they persist beyond short forecast windows.', citations: [{ source: 'Adaptive control in ecological systems', page: 9 }, { source: 'Reproducibility of low-data scientific models', page: 8 }] }],
    gaps: [{ claim: 'A multi-site prospective study could test generalisation, robustness, and long-term performance together.', citations: [{ source: 'Adaptive control in ecological systems', page: 14 }] }],
  },
};
