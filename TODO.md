# TODO: Update Drug Safety Toolkit to use only Gemini API for online search

## Completed Tasks

- [x] Remove all other online search functions (Wikipedia, OpenFDA, PubMed, DuckDuckGo)
- [x] Replace search_online function to use only Gemini API
- [x] Remove unused imports (json, googleapiclient.discovery)
- [x] Remove unused constants (GOOGLE_SEARCH_API_KEY, GOOGLE_SEARCH_CX)
- [x] Remove unused functions (search_wikipedia, search_openfda, search_pubmed, search_duckduckgo, extract_key_medical_terms, extract_drug_names)
- [x] Keep UI interface unchanged
- [x] Keep Gemini API integration for online search

## Next Steps

- [ ] User to provide Gemini API key
- [ ] Set GEMINI_API_KEY environment variable
- [ ] Test the application with Gemini API key
