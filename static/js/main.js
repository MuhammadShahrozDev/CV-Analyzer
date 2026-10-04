document.addEventListener('DOMContentLoaded', () => {
  const fileInputs = document.querySelectorAll('input[type="file"]');
  fileInputs.forEach((input) => {
    input.addEventListener('change', () => {
      input.closest('.field')?.classList.toggle('has-file', Boolean(input.files && input.files.length));
    });
  });

  const copyBtn = document.getElementById('copy-email-btn');
  const regenerateBtn = document.getElementById('regenerate-email-btn');
  const subjectField = document.getElementById('email-subject');
  const bodyField = document.getElementById('email-body');
  const contextField = document.getElementById('email-context');
  const statusEl = document.getElementById('email-status');
  const modeRadios = document.querySelectorAll('input[name="email-type"]');

  const setStatus = (text, isError = false) => {
    if (!statusEl) return;
    statusEl.textContent = text;
    statusEl.style.color = isError ? '#c0392b' : '';
  };

  const getSelectedMode = () => {
    const checked = document.querySelector('input[name="email-type"]:checked');
    return checked ? checked.value : 'individual';
  };

  const fetchEmail = async (mode, vary) => {
    const formData = new URLSearchParams();
    formData.set('email_type', mode);
    formData.set('context', contextField.value);
    formData.set('vary', vary ? 'true' : 'false');

    const response = await fetch('/regenerate-email', {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: formData.toString(),
    });

    if (!response.ok) {
      throw new Error('Request failed');
    }
    return response.json();
  };

  if (copyBtn && subjectField && bodyField) {
    copyBtn.addEventListener('click', async () => {
      const combined = `Subject: ${subjectField.value}\n\n${bodyField.value}`;
      try {
        await navigator.clipboard.writeText(combined);
        setStatus('Copied to clipboard.');
      } catch (err) {
        setStatus('Could not copy automatically - please copy manually.', true);
      }
      setTimeout(() => setStatus(''), 3000);
    });
  }

  if (regenerateBtn && subjectField && bodyField && contextField) {
    regenerateBtn.addEventListener('click', async () => {
      regenerateBtn.disabled = true;
      const originalText = regenerateBtn.textContent;
      regenerateBtn.textContent = 'Regenerating...';
      setStatus('');

      try {
        const data = await fetchEmail(getSelectedMode(), true);
        subjectField.value = data.subject || '';
        bodyField.value = data.body || '';
        setStatus('Draft regenerated.');
      } catch (err) {
        setStatus('Could not regenerate the email. Please try again.', true);
      } finally {
        regenerateBtn.disabled = false;
        regenerateBtn.textContent = originalText;
        setTimeout(() => setStatus(''), 3000);
      }
    });
  }

  if (modeRadios.length && subjectField && bodyField && contextField) {
    modeRadios.forEach((radio) => {
      radio.addEventListener('change', async () => {
        setStatus('Switching...');
        modeRadios.forEach((r) => { r.disabled = true; });

        try {
          const data = await fetchEmail(radio.value, false);
          subjectField.value = data.subject || '';
          bodyField.value = data.body || '';
          setStatus(radio.value === 'team' ? 'Switched to Team / B2B draft.' : 'Switched to Individual draft.');
        } catch (err) {
          setStatus('Could not switch email type. Please try again.', true);
        } finally {
          modeRadios.forEach((r) => { r.disabled = false; });
          setTimeout(() => setStatus(''), 3000);
        }
      });
    });
  }
});