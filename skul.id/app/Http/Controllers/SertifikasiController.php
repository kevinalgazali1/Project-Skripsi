<?php

namespace App\Http\Controllers;



use App\Models\User;
use App\Models\Sertifikasi;
use Illuminate\Support\Str;
use Illuminate\Http\Request;
use App\Models\DaftarSertifikasi;
use App\Exports\SertifikasiExport;
use App\Http\Controllers\Controller;
use Illuminate\Support\Facades\Auth;
use Maatwebsite\Excel\Facades\Excel;
use Illuminate\Support\Facades\Response;
use App\Exports\PesertaSertifikasiExport;
use App\Http\Requests\SertifikasiStoreRequest;

class SertifikasiController extends Controller
{
    /**
     * Display a listing of the resource.
     */
    public function index(Request $request)
    {
        $user = User::with('mitraProfile')->find(Auth::id());

        $query = Sertifikasi::with(['daftarSertifikasis.user'])
            ->withCount('daftarSertifikasis')
            ->where('user_id', $user->id)
            ->orderBy('created_at', 'desc');

        // Filter pencarian
        if ($request->filled('judul')) {
            $query->where('judul_sertifikasi', 'like', '%' . $request->judul . '%');
        }

        if ($request->filled('kota')) {
            $query->where('kota', 'like', '%' . $request->kota . '%');
        }

        if ($request->filled('status')) {
            $query->where('status', $request->status);
        }

        if ($request->filled('tahun_mulai')) {
            $query->whereYear('tanggal_mulai', $request->tahun_mulai);
        }

        $sertifikasiMitraSendiri = $query->paginate(5)->withQueryString()->through(function ($s) {
            $pesertas = collect($s->daftarSertifikasis);

            $s->jumlah_asal_sekolah = $pesertas->pluck('asal_sekolah')->unique()->count();
            $s->jumlah_jurusan = $pesertas->pluck('jurusan')->unique()->count();
            $s->jumlah_kuliah = $pesertas->where('status_saat_ini', 'Kuliah')->count();
            $s->jumlah_bekerja_dan_usaha = $pesertas->whereIn('status_saat_ini', ['Bekerja', 'Wirausaha'])->count();
            $s->jumlah_tidak_bekerja = $pesertas->where('status_saat_ini', 'Tidak Bekerja')->count();

            return $s;
        });

        // Ambil sertifikasi mitra lain (tanpa pagination)
        $sertifikasiMitraLain = Sertifikasi::where('user_id', '!=', $user->id)
            ->latest()
            ->limit(6)
            ->get();

        return view('mitra.sertifikasi', compact(
            'sertifikasiMitraSendiri',
            'sertifikasiMitraLain',
            'user'
        ));
    }


    /**
     * Show the form for creating a new resource.
     */
    public function create()
    {
        //
    }

    /**
     * Store a newly created resource in storage.
     */
    public function store(SertifikasiStoreRequest $request)
    {
        try {
            if ($request->validated()) {
                $user_id = $request->user()->id;
                $mitra = $request->user();

                $file = $request->file('foto_sertifikasi');
                $file_name = $file->hashName(); // nama unik
                $file->move(public_path('storage/assets/sertifikasi'), $file_name);

                Sertifikasi::create(
                    $request->except('foto_sertifikasi') + [
                        'foto_sertifikasi' => 'assets/sertifikasi/' . $file_name,
                        'user_id' => $user_id,
                        'mitra_id' => $mitra->id
                    ]
                );
            }

            return redirect()->back()->with('success', 'Sertifikasi Berhasil Dibuat');
        } catch (\Exception $e) {
            return redirect()->back()->with('danger', 'Sertifikasi Gagal Dibuat');
        }
    }

    /**
     * Display the specified resource.
     */
    public function show(string $id)
    {
        //
    }

    /**
     * Show the form for editing the specified resource.
     */
    public function edit(string $id)
    {
        //
    }

    /**
     * Update the specified resource in storage.
     */
    public function update(Request $request, $id)
    {
        $sertif = Sertifikasi::findOrFail($id);

        $sertif->judul_sertifikasi = $request->judul_sertifikasi;
        $sertif->deskripsi = $request->deskripsi;
        $sertif->tanggal_mulai = $request->tanggal_mulai;
        $sertif->tanggal_selesai = $request->tanggal_selesai;
        $sertif->kota = $request->kota;
        $sertif->tempat = $request->tempat;
        $sertif->biaya = $request->biaya;

        // Penyesuaian jika status == 'Gratis'
        if ($request->status === 'Gratis') {
            $sertif->biaya = 0;
            $sertif->nomor_rekening = null;
        } else {
            $sertif->biaya = $request->biaya;
            $sertif->nomor_rekening = $request->nomor_rekening;
        }

        $sertif->status = $request->status; // pastikan status disimpan juga

        if ($request->hasFile('foto_sertifikasi')) {
            // 🔥 Hapus file lama jika ada
            $oldPath = public_path('storage/' . $sertif->foto_sertifikasi);
            if (file_exists($oldPath)) {
                unlink($oldPath);
            }

            // 📥 Upload file baru
            $file = $request->file('foto_sertifikasi');
            $file_name = $file->hashName();
            $file->move(public_path('storage/assets/sertifikasi'), $file_name);

            // 💾 Simpan path baru
            $sertif->foto_sertifikasi = 'assets/sertifikasi/' . $file_name;
        }

        $sertif->save();

        return redirect()->back()->with('success', 'Sertifikasi berhasil diperbarui.');
    }

    /**
     * Remove the specified resource from storage.
     */
    public function destroy(string $id)
    {
        $sertif = Sertifikasi::findOrFail($id);
        if ($sertif->user_id != Auth::id()) {
            abort(403);
        }
        $sertif->delete();
        return redirect()->back()->with('success', 'Sertifikasi berhasil dihapus.');
    }

    public function getPeserta($id)
    {
        $peserta = DaftarSertifikasi::where('sertifikasi_id', $id)->get([
            'nama_lengkap',
            'email',
            'no_hp',
            'bukti_transfer'
        ]);

        return response()->json($peserta);
    }

    public function exportPesertaExcel($id)
    {
        $sertifikasi = Sertifikasi::findOrFail($id);
        $filename = 'daftar_peserta_' . Str::slug($sertifikasi->judul_sertifikasi) . '.xlsx';

        return Excel::download(new PesertaSertifikasiExport($id), $filename);
    }

    public function exportSertifikasi(Request $request)
    {
        return Excel::download(new SertifikasiExport($request), 'data_sertifikasi.xlsx');
    }
}
